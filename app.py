import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from statistics import mean


MODELS = ('tfidf', 'bm25')


def load_nfcorpus():
    from datasets import load_dataset

    ds = load_dataset("BeIR/nfcorpus", "corpus")
    documents = {row['_id']: row for row in ds['corpus']}
    if len(documents) != 3633:
        raise ValueError('El número de documentos de NFCorpus no coincide con BEIR.')
    return documents


STOPWORDS = frozenset('''a about above after again against all am an and any are as at be
because been before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him himself his how i
if in into is it its itself just me more most my myself of off on once only or other our ours
ourselves out over own same she should so some such than that the their theirs them themselves
then there these they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves'''.split())
TOKEN = re.compile(r'[^\W_]+', re.UNICODE)


def stem(token):
    """Stemming flexivo conservador para plurales regulares del inglés."""
    if len(token) > 4 and token.endswith('ies'):
        return token[:-3] + 'y'
    if len(token) > 4 and token.endswith(('ches', 'shes', 'xes', 'sses', 'zzes')):
        return token[:-2]
    if len(token) > 3 and token.endswith('s') and not token.endswith(('ss', 'us', 'is', 'ics', 'ous')):
        return token[:-1]
    return token


def tokenize(text):
    normalized = unicodedata.normalize('NFKC', text).casefold()
    normalized = re.sub(r"\b([^\W_]+)['’]s\b", r'\1', normalized)
    return [stem(token) for token in TOKEN.findall(normalized)
            if token not in STOPWORDS and (len(token) > 1 or token.isdigit())]


class SearchIndex:
    def __init__(self, documents, k1=1.2, b=0.75):
        self.documents = documents
        self.k1, self.b = k1, b
        self.postings = defaultdict(dict)
        self.lengths = {}
        for doc_id, doc in documents.items():
            text = ' '.join((doc.get('title') or '', doc.get('text') or ''))
            counts = Counter(tokenize(text))
            self.lengths[doc_id] = counts.total()
            for term, frequency in counts.items():
                self.postings[term][doc_id] = frequency
        self.vocabulary = frozenset(self.postings)
        avgdl = mean(self.lengths.values()) if documents else 0
        self.avgdl = avgdl
        self.bm25_norm = {doc_id: k1 * (1 - b + b * length / (avgdl or 1))
                          for doc_id, length in self.lengths.items()}
        self.idf_tfidf = {}
        self.idf_bm25 = {}
        norms_squared = defaultdict(float)
        for term, posting in self.postings.items():
            df = len(posting)
            self.idf_tfidf[term] = 1 + math.log((len(documents) + 1) / (df + 1))
            self.idf_bm25[term] = math.log1p((len(documents) - df + 0.5) / (df + 0.5))
            for doc_id, frequency in posting.items():
                norms_squared[doc_id] += (frequency * self.idf_tfidf[term]) ** 2
        self.norms = {doc_id: math.sqrt(value) for doc_id, value in norms_squared.items()}

    def rank(self, query, model, top_k):
        terms = Counter(term for term in tokenize(query) if term in self.vocabulary)
        scores = defaultdict(float)
        if model == 'tfidf':
            query_weights = {term: frequency * self.idf_tfidf[term]
                             for term, frequency in terms.items()}
            query_norm = math.sqrt(sum(weight ** 2 for weight in query_weights.values()))
            for term, weight in query_weights.items():
                for doc_id, frequency in self.postings[term].items():
                    scores[doc_id] += weight * frequency * self.idf_tfidf[term]
            for doc_id in scores:
                scores[doc_id] /= query_norm * self.norms[doc_id]
        elif model == 'bm25':
            for term in terms:
                for doc_id, frequency in self.postings[term].items():
                    scores[doc_id] += (self.idf_bm25[term] * frequency * (self.k1 + 1)
                                       / (frequency + self.bm25_norm[doc_id]))
        else:
            raise ValueError(f'Modelo desconocido: {model}')
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]

    def search(self, query, model, top_k):
        results = []
        for rank, (doc_id, score) in enumerate(self.rank(query, model, top_k), 1):
            doc = self.documents[doc_id]
            title = doc.get('title') or ''
            text = ' '.join((doc.get('text') or title).split())
            snippet = text[:240].rsplit(' ', 1)[0] + '…' if len(text) > 240 else text
            results.append(dict(rank=rank, document_id=doc_id, score=score,
                                title=title, snippet=snippet))
        return dict(model=model, results=results)


from http import HTTPStatus


def create_app(index):
    def application(environ, start_response):
        def reply(status, body, headers=()):
            data = json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            start_response(f'{status.value} {status.phrase}', [
                ('Content-Type', 'application/json; charset=utf-8'),
                ('Content-Length', str(len(data))), *headers,
            ])
            return [data]

        path = environ.get('PATH_INFO', '')
        method = environ.get('REQUEST_METHOD', '')
        expected_method = {'/': 'GET', '/health': 'GET', '/search': 'POST'}.get(path)
        if expected_method is None:
            return reply(HTTPStatus.NOT_FOUND, {'error': 'Ruta desconocida.'})
        if method != expected_method:
            return reply(HTTPStatus.METHOD_NOT_ALLOWED, {'error': 'Método no permitido.'},
                         [('Allow', expected_method)])
        if path == '/':
            return reply(HTTPStatus.OK, {'corpus': 'NFCorpus', 'endpoints': ['GET /health', 'POST /search'],
                                        'example': {'query': 'plant-based diets', 'model': 'bm25', 'top_k': 10}})
        if path == '/health':
            return reply(HTTPStatus.OK, {'status': 'ok', 'corpus': 'NFCorpus',
                                        'documents': len(index.documents), 'models': list(MODELS)})

        if environ.get('CONTENT_TYPE', '').split(';', 1)[0].strip().lower() != 'application/json':
            return reply(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {'error': 'Envía application/json.'})
        try:
            body_size = int(environ.get('CONTENT_LENGTH') or '0')
            if not 0 < body_size <= 65536:
                return reply(HTTPStatus.BAD_REQUEST, {'error': 'El cuerpo debe ocupar entre 1 y 65536 bytes.'})
            payload = json.loads(environ['wsgi.input'].read(body_size))
        except (ValueError, RecursionError):
            return reply(HTTPStatus.BAD_REQUEST, {'error': 'JSON inválido.'})
        if not isinstance(payload, dict):
            return reply(HTTPStatus.BAD_REQUEST, {'error': 'Se espera un objeto JSON.'})
        query = payload.get('query')
        model = payload.get('model')
        top_k = payload.get('top_k', 10)
        if not isinstance(query, str) or not query.strip():
            return reply(HTTPStatus.BAD_REQUEST, {'error': 'query debe ser texto no vacío.'})
        if model not in MODELS:
            return reply(HTTPStatus.BAD_REQUEST, {'error': 'model debe ser bm25 o tfidf.'})
        if type(top_k) is not int or top_k < 1:
            return reply(HTTPStatus.BAD_REQUEST, {'error': 'top_k debe ser un entero positivo.'})
        return reply(HTTPStatus.OK, index.search(query, model, min(top_k, len(index.documents))))

    return application


def make_application():
    documents = load_nfcorpus()
    return create_app(SearchIndex(documents))


if __name__ == '__main__':
    from wsgiref.simple_server import make_server

    with make_server('127.0.0.1', 7860, make_application()) as server:
        print('NFCorpus disponible en http://127.0.0.1:7860/health', flush=True)
        server.serve_forever()
