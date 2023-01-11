def get_top_n(sim_dict, id_to_title, N=3):
    return [(str(doc_id), id_to_title[doc_id]) for doc_id, score in sorted(sim_dict.items(), key=lambda x: x[1],
                                                                           reverse=True)][:N]
