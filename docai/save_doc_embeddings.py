from docai.database import table_docs
from docai.content_generator import ContentGenerator

def save_doc_embeddings(vocabulary, model, strategy='average'):
    """Saves document embeddings in the database using the provided
    model and vocabulary. This function requires that the model
    supports the get_embedding_doc interface.
    """
    docs = table_docs.get_docs_with_names(['Judgment'])
    content_gen = ContentGenerator(docs)

    for doc, content in zip(docs, content_gen):
        emb = model.get_embedding_doc(content, strategy=strategy)
        table_docs.update_embedding(doc, emb)
