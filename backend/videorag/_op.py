#type: ignore
import re
import json
import openai
import asyncio
import tiktoken
from typing import Union
from collections import Counter, defaultdict
from ._splitter import SeparatorSplitter
from ._utils import (
    logger,
    clean_str,
    compute_mdhash_id,
    decode_tokens_by_tiktoken,
    encode_string_by_tiktoken,
    is_float_regex,
    list_of_list_to_csv,
    pack_user_ass_to_openai_messages,
    split_string_by_multi_markers,
    truncate_list_by_token_size,
)
from .base import (
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    SingleCommunitySchema,
    CommunitySchema,
    TextChunkSchema,
    QueryParam,
)
from .prompt import GRAPH_FIELD_SEP, PROMPTS
from ._videoutil import (
    retrieved_segment_caption_async,
)

def chunking_by_token_size(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = []
        lengths = []
        for start in range(0, len(tokens), max_token_size - overlap_token_size):

            chunk_token.append(tokens[start : start + max_token_size])
            lengths.append(min(max_token_size, len(tokens) - start))

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def chunking_by_video_segments(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    max_token_size=1024,
):
    # make sure each segment is not larger than max_token_size
    for index in range(len(tokens_list)):
        if len(tokens_list[index]) > max_token_size:
            tokens_list[index] = tokens_list[index][:max_token_size]
    
    results = []
    chunk_token = []
    chunk_segment_ids = []
    chunk_order_index = 0
    for index, tokens in enumerate(tokens_list):
        
        if len(chunk_token) + len(tokens) <= max_token_size:
            # add new segment
            chunk_token += tokens.copy()
            chunk_segment_ids.append(doc_keys[index])
        else:
            # save the current chunk
            chunk = tiktoken_model.decode(chunk_token)
            results.append(
                {
                    "tokens": len(chunk_token),
                    "content": chunk.strip(),
                    "chunk_order_index": chunk_order_index,
                    "video_segment_id": chunk_segment_ids,
                }
            )
            # new chunk with current segment as begin
            chunk_token = []
            chunk_segment_ids = []
            chunk_token += tokens.copy()
            chunk_segment_ids.append(doc_keys[index])
            chunk_order_index += 1
    
    # save the last chunk
    if len(chunk_token) > 0:
        chunk = tiktoken_model.decode(chunk_token)
        results.append(
            {
                "tokens": len(chunk_token),
                "content": chunk.strip(),
                "chunk_order_index": chunk_order_index,
                "video_segment_id": chunk_segment_ids,
            }
        )
    
    return results
    
    
def chunking_by_seperators(
    tokens_list: list[list[int]],
    doc_keys,
    tiktoken_model,
    overlap_token_size=128,
    max_token_size=1024,
):

    splitter = SeparatorSplitter(
        separators=[
            tiktoken_model.encode(s) for s in PROMPTS["default_text_separator"]
        ],
        chunk_size=max_token_size,
        chunk_overlap=overlap_token_size,
    )
    results = []
    for index, tokens in enumerate(tokens_list):
        chunk_token = splitter.split_tokens(tokens)
        lengths = [len(c) for c in chunk_token]

        # here somehow tricky, since the whole chunk tokens is list[list[list[int]]] for corpus(doc(chunk)),so it can't be decode entirely
        chunk_token = tiktoken_model.decode_batch(chunk_token)
        for i, chunk in enumerate(chunk_token):

            results.append(
                {
                    "tokens": lengths[i],
                    "content": chunk.strip(),
                    "chunk_order_index": i,
                    "full_doc_id": doc_keys[index],
                }
            )

    return results


def get_chunks(new_videos, chunk_func=chunking_by_video_segments, **chunk_func_params):
    inserting_chunks = {}

    new_videos_list = list(new_videos.keys())
    for video_name in new_videos_list:
        segment_id_list = list(new_videos[video_name].keys())
        docs = [new_videos[video_name][index]["content"] for index in segment_id_list]
        doc_keys = [f'{video_name}_{index}' for index in segment_id_list]

        ENCODER = tiktoken.encoding_for_model("gpt-4o")
        tokens = ENCODER.encode_batch(docs, num_threads=16)
        chunks = chunk_func(
            tokens, doc_keys=doc_keys, tiktoken_model=ENCODER, **chunk_func_params
        )

        for chunk in chunks:
            inserting_chunks.update(
                {compute_mdhash_id(chunk["content"], prefix="chunk-"): chunk}
            )

    return inserting_chunks


async def _handle_entity_relation_summary(
    entity_or_relation_name: str,
    description: str,
    global_config: dict,
) -> str:
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    llm_max_tokens = global_config["llm"]["cheap_model_max_token_size"]
    tiktoken_model_name = global_config["tiktoken_model_name"]
    summary_max_tokens = global_config["entity_summary_to_max_tokens"]

    tokens = encode_string_by_tiktoken(description, model_name=tiktoken_model_name)
    if len(tokens) < summary_max_tokens:  # No need for summary
        return description
    prompt_template = PROMPTS["summarize_entity_descriptions"]
    use_description = decode_tokens_by_tiktoken(
        tokens[:llm_max_tokens], model_name=tiktoken_model_name
    )
    context_base = dict(
        entity_name=entity_or_relation_name,
        description_list=use_description.split(GRAPH_FIELD_SEP),
    )
    use_prompt = prompt_template.format(**context_base)
    logger.debug(f"Trigger summary: {entity_or_relation_name}")
    summary = await use_llm_func(use_prompt, max_tokens=summary_max_tokens)
    return summary


async def _handle_single_entity_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 4 or record_attributes[0] != '"entity"':
        return None
    # add this record as a node in the G
    entity_name = clean_str(record_attributes[1].upper())
    if not entity_name.strip():
        return None
    entity_type = clean_str(record_attributes[2].upper())
    entity_description = clean_str(record_attributes[3])
    entity_source_id = chunk_key
    return dict(
        entity_name=entity_name,
        entity_type=entity_type,
        description=entity_description,
        source_id=entity_source_id,
    )


async def _handle_single_relationship_extraction(
    record_attributes: list[str],
    chunk_key: str,
):
    if len(record_attributes) < 5 or record_attributes[0] != '"relationship"':
        return None
    # add this record as edge
    source = clean_str(record_attributes[1].upper())
    target = clean_str(record_attributes[2].upper())
    edge_description = clean_str(record_attributes[3])
    edge_source_id = chunk_key
    weight = (
        float(record_attributes[-1]) if is_float_regex(record_attributes[-1]) else 1.0
    )
    return dict(
        src_id=source,
        tgt_id=target,
        weight=weight,
        description=edge_description,
        source_id=edge_source_id,
    )


async def _merge_nodes_then_upsert(
    entity_name: str,
    nodes_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_entitiy_types = []
    already_source_ids = []
    already_description = []

    already_node = await knowledge_graph_inst.get_node(entity_name)
    if already_node is not None:
        already_entitiy_types.append(already_node["entity_type"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_node["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_node["description"])

    entity_type = sorted(
        Counter(
            [dp["entity_type"] for dp in nodes_data] + already_entitiy_types
        ).items(),
        key=lambda x: x[1],
        reverse=True,
    )[0][0]
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in nodes_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in nodes_data] + already_source_ids)
    )
    description = await _handle_entity_relation_summary(
        entity_name, description, global_config
    )
    node_data = dict(
        entity_type=entity_type,
        description=description,
        source_id=source_id,
    )
    await knowledge_graph_inst.upsert_node(
        entity_name,
        node_data=node_data,
    )
    node_data["entity_name"] = entity_name
    return node_data


async def _merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    knowledge_graph_inst: BaseGraphStorage,
    global_config: dict,
):
    already_weights = []
    already_source_ids = []
    already_description = []
    already_order = []
    if await knowledge_graph_inst.has_edge(src_id, tgt_id):
        already_edge = await knowledge_graph_inst.get_edge(src_id, tgt_id)
        already_weights.append(already_edge["weight"])
        already_source_ids.extend(
            split_string_by_multi_markers(already_edge["source_id"], [GRAPH_FIELD_SEP])
        )
        already_description.append(already_edge["description"])
        already_order.append(already_edge.get("order", 1))

    # [numberchiffre]: `Relationship.order` is only returned from DSPy's predictions
    order = min([dp.get("order", 1) for dp in edges_data] + already_order)
    weight = sum([dp["weight"] for dp in edges_data] + already_weights)
    description = GRAPH_FIELD_SEP.join(
        sorted(set([dp["description"] for dp in edges_data] + already_description))
    )
    source_id = GRAPH_FIELD_SEP.join(
        set([dp["source_id"] for dp in edges_data] + already_source_ids)
    )
    for need_insert_id in [src_id, tgt_id]:
        if not (await knowledge_graph_inst.has_node(need_insert_id)):
            await knowledge_graph_inst.upsert_node(
                need_insert_id,
                node_data={
                    "source_id": source_id,
                    "description": description,
                    "entity_type": '"UNKNOWN"',
                },
            )
    description = await _handle_entity_relation_summary(
        (src_id, tgt_id), description, global_config
    )
    await knowledge_graph_inst.upsert_edge(
        src_id,
        tgt_id,
        edge_data=dict(
            weight=weight, description=description, source_id=source_id, order=order
        ),
    )
    return_edge_data = dict(
        src_tgt=(src_id, tgt_id),
        description=description,
        weight=weight
    )
    return return_edge_data


async def extract_entities(
    chunks: dict[str, TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
    entity_vdb: BaseVectorStorage,
    global_config: dict,
) -> Union[BaseGraphStorage, None]:
    use_llm_func: callable = global_config["llm"]["best_model_func"]
    entity_extract_max_gleaning = global_config["entity_extract_max_gleaning"]
    
    ordered_chunks = list(chunks.items())

    entity_extract_prompt = PROMPTS["entity_extraction"]
    context_base = dict(
        tuple_delimiter=PROMPTS["DEFAULT_TUPLE_DELIMITER"],
        record_delimiter=PROMPTS["DEFAULT_RECORD_DELIMITER"],
        completion_delimiter=PROMPTS["DEFAULT_COMPLETION_DELIMITER"],
        entity_types=",".join(PROMPTS["DEFAULT_ENTITY_TYPES"]),
    )
    continue_prompt = PROMPTS["entiti_continue_extraction"]
    if_loop_prompt = PROMPTS["entiti_if_loop_extraction"]

    already_processed = 0
    already_entities = 0
    already_relations = 0

    async def _process_single_content(chunk_key_dp: tuple[str, TextChunkSchema]):
        nonlocal already_processed, already_entities, already_relations
        chunk_key = chunk_key_dp[0]
        chunk_dp = chunk_key_dp[1]
        content = chunk_dp["content"]
        hint_prompt = entity_extract_prompt.format(**context_base, input_text=content)
        final_result = await use_llm_func(hint_prompt)

        history = pack_user_ass_to_openai_messages(hint_prompt, final_result)
        for now_glean_index in range(entity_extract_max_gleaning):
            glean_result = await use_llm_func(continue_prompt, history_messages=history)

            history += pack_user_ass_to_openai_messages(continue_prompt, glean_result)
            final_result += glean_result
            if now_glean_index == entity_extract_max_gleaning - 1:
                break

            if_loop_result: str = await use_llm_func(if_loop_prompt, history_messages=history)
            if_loop_result = if_loop_result.strip().strip('"').strip("'").lower()
            if if_loop_result != "yes":
                break

        records = split_string_by_multi_markers(
            final_result,
            [context_base["record_delimiter"], context_base["completion_delimiter"]],
        )

        maybe_nodes = defaultdict(list)
        maybe_edges = defaultdict(list)
        for record in records:
            record = re.search(r"\((.*)\)", record)
            if record is None:
                continue
            record = record.group(1)
            record_attributes = split_string_by_multi_markers(
                record, [context_base["tuple_delimiter"]]
            )
            if_entities = await _handle_single_entity_extraction(
                record_attributes, chunk_key
            )
            if if_entities is not None:
                maybe_nodes[if_entities["entity_name"]].append(if_entities)
                continue

            if_relation = await _handle_single_relationship_extraction(
                record_attributes, chunk_key
            )
            if if_relation is not None:
                maybe_edges[(if_relation["src_id"], if_relation["tgt_id"])].append(
                    if_relation
                )
        already_processed += 1
        already_entities += len(maybe_nodes)
        already_relations += len(maybe_edges)
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        logger.info(f"{now_ticks} Processed {already_processed} chunks, {already_entities} entities(duplicated), {already_relations} relations(duplicated)\r")
        return dict(maybe_nodes), dict(maybe_edges)

    # use_llm_func is wrapped in ascynio.Semaphore, limiting max_async callings
    results = await asyncio.gather(
        *[_process_single_content(c) for c in ordered_chunks]
    )
    maybe_nodes = defaultdict(list)
    maybe_edges = defaultdict(list)
    for m_nodes, m_edges in results:
        for k, v in m_nodes.items():
            maybe_nodes[k].extend(v)
        for k, v in m_edges.items():
            # it's undirected graph
            maybe_edges[tuple(sorted(k))].extend(v)
    all_entities_data = await asyncio.gather(
        *[
            _merge_nodes_then_upsert(k, v, knowledge_graph_inst, global_config)
            for k, v in maybe_nodes.items()
        ]
    )
    all_edges_data = await asyncio.gather(
        *[
            _merge_edges_then_upsert(k[0], k[1], v, knowledge_graph_inst, global_config)
            for k, v in maybe_edges.items()
        ]
    )
    if not len(all_entities_data):
        logger.warning("Didn't extract any entities, maybe your LLM is not working")
        return None
    if entity_vdb is not None:
        data_for_vdb = {
            compute_mdhash_id(dp["entity_name"], prefix="ent-"): {
                "content": dp["entity_name"] + dp["description"],
                "entity_name": dp["entity_name"],
            }
            for dp in all_entities_data
        }
        await entity_vdb.upsert(data_for_vdb)
    return knowledge_graph_inst, all_entities_data, all_edges_data


async def _find_most_related_segments_from_entities(
    topk_chunks: int,
    node_datas: list[dict],
    text_chunks_db: BaseKVStorage[TextChunkSchema],
    knowledge_graph_inst: BaseGraphStorage,
):
    text_units = [
        split_string_by_multi_markers(dp["source_id"], [GRAPH_FIELD_SEP])
        for dp in node_datas
    ]
    edges = await asyncio.gather(
        *[knowledge_graph_inst.get_node_edges(dp["entity_name"]) for dp in node_datas]
    )
    all_one_hop_nodes = set()
    for this_edges in edges:
        if not this_edges:
            continue
        all_one_hop_nodes.update([e[1] for e in this_edges])
    all_one_hop_nodes = list(all_one_hop_nodes)
    all_one_hop_nodes_data = await asyncio.gather(
        *[knowledge_graph_inst.get_node(e) for e in all_one_hop_nodes]
    )
    all_one_hop_text_units_lookup = {
        k: set(split_string_by_multi_markers(v["source_id"], [GRAPH_FIELD_SEP]))
        for k, v in zip(all_one_hop_nodes, all_one_hop_nodes_data)
        if v is not None
    }
    all_text_units_lookup = {}
    for index, (this_text_units, this_edges) in enumerate(zip(text_units, edges)):
        for c_id in this_text_units:
            if c_id in all_text_units_lookup:
                continue
            relation_counts = 0
            for e in this_edges:
                if (
                    e[1] in all_one_hop_text_units_lookup
                    and c_id in all_one_hop_text_units_lookup[e[1]]
                ):
                    relation_counts += 1
            all_text_units_lookup[c_id] = {
                "data": await text_chunks_db.get_by_id(c_id),
                "order": index,
                "relation_counts": relation_counts,
            }
    if any([v is None for v in all_text_units_lookup.values()]):
        logger.warning("Text chunks are missing, maybe the storage is damaged")
    all_text_units = [
        {"id": k, **v} for k, v in all_text_units_lookup.items() if v is not None
    ]
    sorted_text_units = sorted(
        all_text_units, key=lambda x: -x["relation_counts"]
    )[:topk_chunks]
    
    chunk_related_segments = set()
    for _chunk_data in sorted_text_units:
        for s_id in _chunk_data['data']['video_segment_id']:
            chunk_related_segments.add(s_id)
    
    return chunk_related_segments

async def _refine_entity_retrieval_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    query_rewrite_prompt = PROMPTS["query_rewrite_for_entity_retrieval"]
    query_rewrite_prompt = query_rewrite_prompt.format(input_text=query)
    final_result = await use_llm_func(query_rewrite_prompt)
    return final_result

async def _refine_visual_retrieval_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    query_rewrite_prompt = PROMPTS["query_rewrite_for_visual_retrieval"]
    query_rewrite_prompt = query_rewrite_prompt.format(input_text=query)
    final_result = await use_llm_func(query_rewrite_prompt)
    return final_result

async def _extract_keywords_query(
    query,
    query_param: QueryParam,
    global_config: dict,
):
    use_llm_func: callable = global_config["llm"]["cheap_model_func"]
    keywords_prompt = PROMPTS["keywords_extraction"]
    keywords_prompt = keywords_prompt.format(input_text=query)
    final_result = await use_llm_func(keywords_prompt)
    return final_result

async def _retrieve_and_filter_segments(
    query,
    entities_vdb,
    text_chunks_db,
    chunks_vdb,
    video_segments,
    video_segment_feature_vdb,
    knowledge_graph_inst,
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["llm"]["best_model_func"]
    
    # naive chunks
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    chunks_ids = [r["id"] for r in results] if results else []
    chunks = await text_chunks_db.get_by_ids(chunks_ids)

    maybe_trun_chunks = truncate_list_by_token_size(
        chunks,
        key=lambda x: x["content"],
        max_token_size=query_param.naive_max_token_for_text_unit,
    )
    logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
    section = "-----New Chunk-----\n".join([c["content"] for c in maybe_trun_chunks])
    retreived_chunk_context = section if maybe_trun_chunks else "No Content"
    
    # Gather text segments with metadata
    text_segment_objects = []
    for c in maybe_trun_chunks:
        if "video_segment_id" in c:
            for s_id in c["video_segment_id"]:
                text_segment_objects.append({"id": s_id, "type": "text", "score": 10.0})
    
    # Entity retrieval
    query_for_entity_retrieval = await _refine_entity_retrieval_query(
        query, query_param, global_config
    )
    entity_results = await entities_vdb.query(query_for_entity_retrieval, top_k=query_param.top_k)
    entity_retrieved_objects = []
    
    entity_retrieved_ids = set()
    if len(entity_results):
        node_datas = await asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entity_results]
        )
        # Filter None
        node_datas = [n for n in node_datas if n is not None]
        
        # We need to re-match with entity_results but index might shift if some are None?
        # Safe way:
        # Re-fetch only successful ones or just map carefully.
        # Original code used zip but filtered None.
        # Assuming get_node returns None if missing.
        
        # Simplified for safety:
        # Just use what we have.
        
        # Recalculate node degrees for valid nodes only?
        # Original logic was complex with zip. I'll simplify relying on `_find_most_related...`
        # Using a safer pattern:
        valid_nodes = []
        for r in entity_results:
            n = await knowledge_graph_inst.get_node(r["entity_name"])
            if n:
                 d = await knowledge_graph_inst.node_degree(r["entity_name"])
                 valid_nodes.append({**n, "entity_name": r["entity_name"], "rank": d})

        entity_retrieved_ids = entity_retrieved_ids.union(await _find_most_related_segments_from_entities(
            global_config["retrieval_topk_chunks"], valid_nodes, text_chunks_db, knowledge_graph_inst
        ))
        for eid in entity_retrieved_ids:
             entity_retrieved_objects.append({"id": eid, "type": "entity", "score": 2.0})

    # Visual retrieval
    query_for_visual_retrieval = await _refine_visual_retrieval_query(
        query, query_param, global_config
    )
    segment_results = await video_segment_feature_vdb.query(query_for_visual_retrieval)
    visual_retrieved_objects = []
    if len(segment_results):
        for n in segment_results:
            visual_retrieved_objects.append({"id": n['__id__'], "type": "visual", "score": n.get('distance', 0.0)})
    
    # Consolidate and Deduplicate
    all_retrieved_objects = text_segment_objects + entity_retrieved_objects + visual_retrieved_objects
    unique_segments = {}
    for obj in all_retrieved_objects:
        sid = obj["id"]
        if sid not in unique_segments:
            unique_segments[sid] = obj
        else:
            existing = unique_segments[sid]
            if obj["type"] == "visual":
                unique_segments[sid] = obj
            elif obj["type"] == existing["type"] and obj["score"] > existing["score"]:
                unique_segments[sid] = obj
                
    retrieved_segments = list(unique_segments.values())
    
    # Filtering
    already_processed = 0
    async def _filter_single_segment(knowledge: str, segment_key_dp: tuple[str, str]):
        nonlocal use_model_func, already_processed
        segment_key = segment_key_dp[0]
        segment_content = segment_key_dp[1]
        filter_prompt = PROMPTS["filtering_segment"]
        filter_prompt = filter_prompt.format(caption=segment_content, knowledge=knowledge)
        result = await use_model_func(filter_prompt, global_config=global_config)
        already_processed += 1
        return (segment_key, result)
    
    rough_captions = {}
    for seg_obj in retrieved_segments:
        s_id = seg_obj["id"]
        # Safe split
        parts = s_id.split('_')
        if len(parts) < 2: continue
        video_name = '_'.join(parts[:-1])
        index = parts[-1]
        try:
            rough_captions[s_id] = video_segments._data[video_name][index]["content"]
        except:
            continue

    results = await asyncio.gather(
        *[_filter_single_segment(query, (s_id, rough_captions[s_id])) for s_id in rough_captions]
    )
    
    remain_segment_ids = [x[0] for x in results if 'yes' in x[1].lower()]
    remain_segments = [unique_segments[rid] for rid in remain_segment_ids if rid in unique_segments]
    
    
    if len(remain_segments) == 0:
        if global_config.get("strict_filtering", False):
            # For precise clips, returning nothing is better than garbage
            return [], retreived_chunk_context
        remain_segments = retrieved_segments # Fallback
        
    return remain_segments, retreived_chunk_context


async def videorag_query(
    query,
    entities_vdb,
    text_chunks_db,
    chunks_vdb,
    video_path_db,
    video_segments,
    video_segment_feature_vdb,
    knowledge_graph_inst,
    query_param: QueryParam,
    global_config: dict,
):
    use_model_func = global_config["llm"]["best_model_func"]
    
    # 1. Retrieve & Filter
    remain_segments, retreived_chunk_context = await _retrieve_and_filter_segments(
        query, entities_vdb, text_chunks_db, chunks_vdb, video_segments, 
        video_segment_feature_vdb, knowledge_graph_inst, query_param, global_config
    )
    
    # 2. Captioning (Hybrid)
    keywords_for_caption = await _extract_keywords_query(query, query_param, global_config)
    caption_results = await retrieved_segment_caption_async(
        keywords_for_caption,
        remain_segments,
        video_path_db,
        video_segments,
        num_sampled_frames=global_config['fine_num_frames_per_segment'],
        global_config=global_config,
    )

    # 3. Form Context
    text_units_section_list = [["video_name", "start_time", "end_time", "content"]]
    for s_id in caption_results:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        try:
            time_str = video_segments._data[video_name][index]["time"]
            start_time = eval(time_str.split('-')[0])
            end_time = eval(time_str.split('-')[1])
            start_fmt = f"{start_time // 3600}:{(start_time % 3600) // 60}:{start_time % 60}"
            end_fmt = f"{end_time // 3600}:{(end_time % 3600) // 60}:{end_time % 60}"
            text_units_section_list.append([video_name, start_fmt, end_fmt, caption_results[s_id]])
        except: continue
        
    text_units_context = list_of_list_to_csv(text_units_section_list)
    retreived_video_context = f"\n-----Retrieved Knowledge From Videos-----\n```csv\n{text_units_context}\n```\n"
    
    # 4. Generate Answer (Text Only)
    sys_prompt_temp = PROMPTS["videorag_response_wo_reference"] if query_param.wo_reference else PROMPTS["videorag_response"]
    sys_prompt = sys_prompt_temp.format(
        video_data=retreived_video_context,
        chunk_data=retreived_chunk_context,
        response_type=query_param.response_type
    )
    
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        global_config=global_config,
    )
    
    # Return both response and sources
    # Sources are 'remain_segments' which contain metadata
    return response, remain_segments


async def search_precise_clips(
    query,
    entities_vdb,
    text_chunks_db,
    chunks_vdb,
    video_segments,
    video_segment_feature_vdb,
    knowledge_graph_inst,
    query_param: QueryParam,
    global_config: dict,
):
    """
    Find relevant clips with PRECISE sentence-level alignment.
    """
    from ._utils import limit_async_func_call
    # Upgrade to Best Model for higher precision (slower but worth it)
    use_model_func = global_config["llm"]["best_model_func"] 
    
    # 1. Retrieve & Filter (Reuse logic)
    # Enable strict filtering to avoid fallbacks
    filtered_config = global_config.copy()
    filtered_config["strict_filtering"] = True
    
    remain_segments, _ = await _retrieve_and_filter_segments(
        query, entities_vdb, text_chunks_db, chunks_vdb, video_segments, 
        video_segment_feature_vdb, knowledge_graph_inst, query_param, filtered_config
    )
    
    if not remain_segments:
        return []

    # 2. Extract Precise Sentences
    clips = []
    
    # Limit processing to top 5 segments for speed
    top_segments = sorted(remain_segments, key=lambda x: x.get('score', 0), reverse=True)[:5]
    
    async def process_precise_clip(seg):
        s_id = seg["id"]
        parts = s_id.split('_')
        video_name = '_'.join(parts[:-1])
        index = parts[-1]
        
        segment_info = video_segments._data[video_name][index]
        detailed_transcript = segment_info.get("detailed_transcript")
        
        # Default/Fallback time (30s)
        time_range = segment_info["time"]
        try:
            # Handle float or int strings properly
            seg_start_offset = float(time_range.split('-')[0])
            seg_end_offset = float(time_range.split('-')[1])
        except:
             # Fallback if parsing fails
             seg_start_offset = 0.0
             seg_end_offset = 30.0

        start_sec = seg_start_offset
        end_sec = seg_end_offset
        
        caption = segment_info["transcript"] # Default caption
        speaker_id = None

        if detailed_transcript and detailed_transcript.get("sentences"):
            # Use LLM to pick relevant sentences
            sentences = detailed_transcript["sentences"]
            # Format identifying list
            sent_list_str = ""
            for i, s in enumerate(sentences):
                spk = s.get("speaker_id", "?")
                sent_list_str += f"{i} [Spk {spk}]: {s['text']}\n"
                
            prompt = f"""Query/Context: {query}

Transcript Sentences:
{sent_list_str}

Identify the indices of the sentences that are relevant to the query.
- Select continuous sentences if they form a complete thought or speech.
- If the query implies a specific speaker's speech, select the full relevant segment for that speaker.
- If unsure, select the sentences containing the keywords from the query.

Return *only* a JSON object:
{{
  "reasoning": "explanation",
  "indices": [int]
}}
"""
            
            try:
                # Use JSON Mode
                res = await use_model_func(prompt, response_format={"type": "json_object"})
                import json
                
                indices = []
                try:
                    data = json.loads(res)
                    indices = data.get("indices", [])
                except Exception as e:
                    logger.warning(f"Failed to parse JSON for {s_id}: {e}")
                    indices = []
                
                if indices:
                    # Find min start and max end
                    relevant_sents = [sentences[i] for i in indices if i < len(sentences)]
                    if relevant_sents:
                        # Snap to start/end relative to segment + segment offset
                        # Timestamps in 'start'/'end' are ms relative to segment start
                        rel_start = relevant_sents[0]["start"] / 1000.0 # ms -> s
                        rel_end = relevant_sents[-1]["end"] / 1000.0 # ms -> s
                        
                        start_sec = seg_start_offset + rel_start
                        end_sec = seg_start_offset + rel_end
                        
                        caption = " ".join([s["text"] for s in relevant_sents])
                        
                        # Determine dominant speaker
                        speakers = [s.get("speaker_id") for s in relevant_sents if s.get("speaker_id") is not None]
                        if speakers:
                            # Use Counter to find most common speaker
                            from collections import Counter
                            speaker_id = Counter(speakers).most_common(1)[0][0]
                else:
                    # Strict Mode: If we have transcripts but LLM found NOTHING relevant,
                    # DO NOT include this segment (avoid irrelevant fallbacks).
                    # Exception: If specific keywords match? No, rely on LLM.
                    return None

            except Exception as e:
                logger.warning(f"Failed precise extraction for {s_id}: {e}")
                # If LLM fails (exception), do we keep it? 
                # Safer to discard to avoid noise, or keep as fallback?
                # Let's keep as fallback if exception occurs (safe).
                pass

        return {
            "video_id": video_name,
            "start": float(start_sec),
            "end": float(end_sec),
            "caption": caption,
            "score": float(seg.get('score', 0)),
            "speaker_id": speaker_id
        }

    results = await asyncio.gather(*[process_precise_clip(seg) for seg in top_segments])
    raw_clips = [r for r in results if r]
    
    if not raw_clips:
        return []

    # 3. Merge Adjacent/Overlapping Clips and Apply Padding
    # Sort by video_id then start time
    raw_clips.sort(key=lambda x: (x["video_id"], x["start"]))
    
    merged_clips = []
    current_clip = raw_clips[0]
    
    # 0.5s padding
    PADDING = 0.5
    DEFAULT_MERGE_THRESHOLD = 2.0 # Allow 2s gap to simulate continuous speech since we lack speaker_id
    
    # Apply initial padding to first clip match (start only for now, end is dynamic)
    current_clip["start"] = max(0.0, current_clip["start"] - PADDING)
    
    for next_clip in raw_clips[1:]:
        # Check if same video
        if next_clip["video_id"] == current_clip["video_id"]:
            
            gap = next_clip["start"] - current_clip["end"]
            
            # Since we cannot detect speakers reliably in this region (Singpore/International),
            # we will assume clips within a short time window (2.0s) belong to the same flow/speaker
            # and merge them to create a continuous clip.
            threshold = DEFAULT_MERGE_THRESHOLD
            
            if gap <= threshold:
                # Merge
                current_clip["end"] = max(current_clip["end"], next_clip["end"])
                # Merge captions if not duplicate
                if next_clip["caption"] not in current_clip["caption"]:
                    current_clip["caption"] += " " + next_clip["caption"]
                # Keep highest score
                current_clip["score"] = max(current_clip["score"], next_clip["score"])
                continue

        # Finalize current clip
        current_clip["end"] += PADDING
        merged_clips.append(current_clip)
        
        # Start new clip
        current_clip = next_clip
        current_clip["start"] = max(0.0, current_clip["start"] - PADDING)
        
    # Append last
    current_clip["end"] += PADDING
    merged_clips.append(current_clip)
    
    # Sort by score for final presentation
    merged_clips.sort(key=lambda x: x["score"], reverse=True)
    
    return merged_clips


async def videorag_query_multiple_choice(
    query,
    entities_vdb,
    text_chunks_db,
    chunks_vdb,
    video_path_db,
    video_segments,
    video_segment_feature_vdb,
    knowledge_graph_inst,
    query_param: QueryParam,
    global_config: dict,
) -> str:
    """_summary_
    A copy of the videorag_query function with several updates for handling multiple-choice queries.
    """
    use_model_func = global_config["llm"]["best_model_func"]
    query = query
    
    # naive chunks
    results = await chunks_vdb.query(query, top_k=query_param.top_k)
    # NOTE: I update here, not len results can also process
    if len(results):
        chunks_ids = [r["id"] for r in results]
        chunks = await text_chunks_db.get_by_ids(chunks_ids)

        maybe_trun_chunks = truncate_list_by_token_size(
            chunks,
            key=lambda x: x["content"],
            max_token_size=query_param.naive_max_token_for_text_unit,
        )
        logger.info(f"Truncate {len(chunks)} to {len(maybe_trun_chunks)} chunks")
        section = "-----New Chunk-----\n".join([c["content"] for c in maybe_trun_chunks])
        retreived_chunk_context = section
    else:
        retreived_chunk_context = "No Content"
        
    # visual retrieval
    query_for_entity_retrieval = await _refine_entity_retrieval_query(
        query,
        query_param,
        global_config,
    )
    entity_results = await entities_vdb.query(query_for_entity_retrieval, top_k=query_param.top_k)
    entity_retrieved_segments = set()
    if len(entity_results):
        node_datas = await asyncio.gather(
            *[knowledge_graph_inst.get_node(r["entity_name"]) for r in entity_results]
        )
        if not all([n is not None for n in node_datas]):
            logger.warning("Some nodes are missing, maybe the storage is damaged")
        node_degrees = await asyncio.gather(
            *[knowledge_graph_inst.node_degree(r["entity_name"]) for r in entity_results]
        )
        node_datas = [
            {**n, "entity_name": k["entity_name"], "rank": d}
            for k, n, d in zip(entity_results, node_datas, node_degrees)
            if n is not None
        ]
        entity_retrieved_segments = entity_retrieved_segments.union(await _find_most_related_segments_from_entities(
            global_config["retrieval_topk_chunks"], node_datas, text_chunks_db, knowledge_graph_inst
        ))
    
    # visual retrieval
    query_for_visual_retrieval = await _refine_visual_retrieval_query(
        query,
        query_param,
        global_config,
    )
    segment_results = await video_segment_feature_vdb.query(query_for_visual_retrieval)
    visual_retrieved_segments = set()
    if len(segment_results):
        for n in segment_results:
            visual_retrieved_segments.add(n['__id__'])
    
    # caption
    retrieved_segments = list(entity_retrieved_segments.union(visual_retrieved_segments))
    retrieved_segments = sorted(
        retrieved_segments,
        key=lambda x: (
            '_'.join(x.split('_')[:-1]), # video_name
            eval(x.split('_')[-1]) # index
        )
    )
    logger.info(query_for_entity_retrieval)
    logger.info(f"Retrieved Text Segments {entity_retrieved_segments}")
    logger.info(query_for_visual_retrieval)
    logger.info(f"Retrieved Visual Segments {visual_retrieved_segments}")
    
    already_processed = 0
    async def _filter_single_segment(knowledge: str, segment_key_dp: tuple[str, str]):
        nonlocal use_model_func, already_processed
        segment_key = segment_key_dp[0]
        segment_content = segment_key_dp[1]
        filter_prompt = PROMPTS["filtering_segment"]
        filter_prompt = filter_prompt.format(caption=segment_content, knowledge=knowledge)
        result = await use_model_func(filter_prompt, global_config=global_config)
        already_processed += 1
        now_ticks = PROMPTS["process_tickers"][
            already_processed % len(PROMPTS["process_tickers"])
        ]
        logger.info(f"{now_ticks} Checked {already_processed} segments\r")
        return (segment_key, result)
    
    rough_captions = {}
    for s_id in retrieved_segments:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        rough_captions[s_id] = video_segments._data[video_name][index]["content"]
    results = await asyncio.gather(
        *[_filter_single_segment(query, (s_id, rough_captions[s_id])) for s_id in rough_captions]
    )
    remain_segments = [x[0] for x in results if 'yes' in x[1].lower()]
    logger.info(f"{len(remain_segments)} Video Segments remain after filtering")
    if len(remain_segments) == 0:
        logger.info("Since no segments remain after filtering, we utilized all the retrieved segments.")
        remain_segments = retrieved_segments
    logger.info(f"Remain segments {remain_segments}")
    
    # visual retrieval
    keywords_for_caption = await _extract_keywords_query(
        query,
        query_param,
        global_config,
    )
    logger.info(f"Keywords: {keywords_for_caption}")
    caption_results = await retrieved_segment_caption_async(
        keywords_for_caption,
        remain_segments,
        video_path_db,
        video_segments,
        num_sampled_frames=global_config['fine_num_frames_per_segment'],
        global_config=global_config,
    )

    ## data table
    text_units_section_list = [["video_name", "start_time", "end_time", "content"]]
    for s_id in caption_results:
        video_name = '_'.join(s_id.split('_')[:-1])
        index = s_id.split('_')[-1]
        start_time = eval(video_segments._data[video_name][index]["time"].split('-')[0])
        end_time = eval(video_segments._data[video_name][index]["time"].split('-')[1])
        start_time = f"{start_time // 3600}:{(start_time % 3600) // 60}:{start_time % 60}"
        end_time = f"{end_time // 3600}:{(end_time % 3600) // 60}:{end_time % 60}"
        text_units_section_list.append([video_name, start_time, end_time, caption_results[s_id]])
    text_units_context = list_of_list_to_csv(text_units_section_list)

    retreived_video_context = f"\n-----Retrieved Knowledge From Videos-----\n```csv\n{text_units_context}\n```\n"
    
    # NOTE: I update here to use a different prompt
    sys_prompt_temp = PROMPTS["videorag_response_for_multiple_choice_question"]
        
    sys_prompt = sys_prompt_temp.format(
        video_data=retreived_video_context,
        chunk_data=retreived_chunk_context,
        response_type=query_param.response_type
    )
    response = await use_model_func(
        query,
        system_prompt=sys_prompt,
        use_cache=False,
    )
    while True:
        try:
            json_response = json.loads(response)
            assert "Answer" in json_response and "Explanation" in json_response
            return json_response
        except Exception as e:
            logger.info(f"Response is not valid JSON for query {query}. Found {e}. Retrying...")
            response = await use_model_func(
                query,
                system_prompt=sys_prompt,
                use_cache=False,
            )
    