def _score(self, query, doc_id):
    """
    This function calculate the bm25 score for given query and document.

    Parameters:
    -----------
    query: list of token representing the query. For example: ['look', 'blue', 'sky']
    doc_id: integer, document id.

    Returns:
    -----------
    score: float, bm25 score.
    """
    score = 0.0
    doc_len = DL[str(doc_id)]

    for term in query:
        if term in self.index.term_total.keys():
            term_frequencies = dict(self.pls[self.words.index(term)])
            if doc_id in term_frequencies.keys():
                freq = term_frequencies[doc_id]
                numerator = self.idf[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.AVGDL)
                score += (numerator / denominator)
    return score