import torch


def save_trec(rst_file, rst_dict):
    with open(rst_file, 'w') as writer:
        for q_id, scores in rst_dict.items():
            res = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)
            for rank, value in enumerate(res):
                writer.write(q_id + ' Q0 ' + str(value[0]) + ' ' + str(rank + 1) + ' ' + str(value[1][0]) + ' DREQ\n')


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def save_checkpoint(save_path, model):
    if save_path is None:
        return

    torch.save(model.state_dict(), save_path)
    print(f'Model saved to ==> {save_path}')


def load_checkpoint(load_path, model, device):
    """
    Load a checkpoint written by save_checkpoint.

    save_checkpoint writes a bare state_dict. The previous version of this
    function indexed state_dict['model_state_dict'], a key that is never
    written, so it raised KeyError on any checkpoint produced by this code.
    Both layouts are accepted here so that checkpoints saved either way load.
    """
    if load_path is None:
        return

    state_dict = torch.load(load_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    print(f'Model loaded from <== {load_path}')