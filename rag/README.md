# Retrieval Augmented Generation (RAG) Overview

RAG receives the promtp and the query from the user. 
Instead of asking LLM for this request, it augments the prompt with relevant context information.
The query is sent to a data source (e.g. DB, multiple embedding spaces), to retrieve the relevant data for the query.
The prompt + query are augmented with the data/context which are then sent to the LLM for it to produce a more informed answer.

## Important concepts
- Prompt: Implemented as a request/ question
- Index: Organize data in order to search and provide context for the `Retrieval` part of RAG
  Involves: load, split, embed, store into DB
- Retrieval: Recover information based on the request
  - Question/Answering

## Prompt
Can be seen as: Query (injected in a) + Prompt + (augmented with) Context + (to get a response from) LLM

(Base) Prompt (indicate the agent profile, behaviour, response, etc.)
    e.g. <br>
        ```<<SYS>> You are a professional xxx providing helpful xxx. 
        Your answers should include xxx. 
        Answer the following questions using information from xxx  
        CONTEXT_INFORMATION: ...
        QUESTION: ...```

Query: (question) "What is the total xxx?"
        <br>Inject in prompt

Context: (e.g., from DB) Use query information for data collection. 
        <br>Augment prompt with data

LLM: Process prompt

### Llama example

```
Context information from various sources is below.
---------
{context_str}
---------
Given the information from various sources and not prior knowledge, answer the query.
Query: {query_str}
Answer:
```

<br>Context can also be a list of options:
`{context_list}` Then query could be something like `return the choice that is most relevant to the question`
<br>Answer could be required to provide justification:
`{answer} Explain why this choice was selected.` 

## Indexing - Context preparation 

Load -> Split -> Embed -> Store
1. Load (e.g., unstructured documents - might have metadata available)
   - LlamaHub data loaders (pdf, csv, docx, html, image, ..)
2. Split / chunks
    - Tokenizer (e.g., word level, character level)
3. Embed
    - Choose an embedding model (model size -million parameters-, memory it uses, context window -tokens-, etc)
    - https://huggingface.co/spaces/mteb/leaderboard
4. Store (e.g., Data chunk + Embeddings + metadata)
    - Vector database (verify that handles embedding dimensions from embedding-model)
    - Choose: do they provide sharding -database partitioning-? are they scalable?
    - https://superlinked.com/vector-db-comparison

## Retrieval

### SQL - retrieve by similarity

Similarity - dot product:<br>
`query embedding: [QE]`<br>
`SELECT post, emb<*>[QE] AS score
FROM posts
ORDER BY xxx DESC`

### High-level API - <a href='https://superlinked.com/' target='_blank'>Superlinked</a>:

```
    query = Query(post_index)
    .find(post)
    .similar(relevance_space.text, Param(XXX))
    
    app.query(query, XXX="what is x>")
```
Will be augmenting prompt for RAG.
Additional options might be available: weighted processing, recency, output formatting, etc.


## Related topics: RankGPT

See <a href='https://github.com/sunnweiwei/RankGPT' target='_blank'>github.com/sunnweiwei/RankGPT</a>

The user states the query, this query is sent to the DB to collect relevant context/data.

Before augmenting the prompt:<br>
we ask the LLM to rerank the retrieved context/data based on the relevance to the query.

Then we augment the prompt with the reranked context and make the full request to the LLM.

## Related topics
### Multi-query Retrieval
See <a href='https://js.langchain.com/v0.1/docs/modules/data_connection/retrievers/multi-query-retriever/' target="_blank">
js.langchain.com/v0.1/docs/modules/data_connection/retrievers/multi-query-retriever/</a>

Instead of sending the query to the DB, we ask first the LLM to provide similar questions to the original query.

Then these multiple queries are sent to the DB, to extend the context.

The whole list of resulting context is then used to augment the prompt and query.

### Contextual Compression
See <a href='https://js.langchain.com/v0.1/docs/modules/data_connection/retrievers/contextual_compression/' target='_blank'>
js.langchain.com/v0.1/docs/modules/data_connection/retrievers/contextual_compression/</a>

In some cases, the documents retrieved for the context extraction include relevant information, but include as well non-relevant information.
Contextual compression attempts to remove the non-relevant parts of the retrieved documents, before augmenting the prompt and query.

### Hypothetical Document Embeddings (HyDE)
See <a href='https://aclanthology.org/2023.acl-long.99/' target='_blank'>
Precise Zero-Shot Dense Retrieval without Relevance Labels</a>

The idea behing HyDE is to hypothesize an extended form of the query: 
ask the LLM to hypothesize an answer (hypothetical document) for the query.

Then, we could use hypothetical document to perform similarity search and identify context/ data documents. 
The identified documents can then augment the prompt + query and perform the final answer retrieval.


