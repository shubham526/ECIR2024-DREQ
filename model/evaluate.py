import torch
from tqdm import tqdm


def evaluate(model, data_loader, device):
    rst_dict = {}
    model.eval()

    num_batch = len(data_loader)

    with torch.no_grad():
        for batch in tqdm(data_loader, total=num_batch):
            query_id, doc_id, label = batch['query_id'], batch['doc_id'], batch['label']

            batch_score, _ = model(
                query_input_ids=batch['query_input_ids'].to(device),
                query_attention_mask=batch['query_attention_mask'].to(device),
                query_token_type_ids=batch['query_token_type_ids'].to(device),
                doc_text_emb=batch['doc_text_emb'].to(device),
                doc_entity_emb=batch['doc_entity_emb'].to(device),
            )

            batch_score = batch_score.detach().cpu().tolist()
            for (q_id, d_id, b_s, l) in zip(query_id, doc_id, batch_score, label):
                if q_id not in rst_dict:
                    rst_dict[q_id] = {}
                if d_id not in rst_dict[q_id] or b_s > rst_dict[q_id][d_id][0]:
                    rst_dict[q_id][d_id] = [b_s, l]

    return rst_dict
