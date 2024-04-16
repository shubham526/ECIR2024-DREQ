import tqdm


class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, metric, data_loader, use_cuda, device):
        self._model = model
        self._optimizer = optimizer
        self._criterion = criterion
        self._scheduler = scheduler
        self._metric = metric
        self._data_loader = data_loader
        self._use_cuda = use_cuda
        self._device = device

    def make_train_step(self):
        # Builds function that performs a step in the train loop

        def train_step(train_batch):
            # Sets model to TRAIN mode
            self._model.train()

            # Zero the gradients
            self._optimizer.zero_grad()

            # Makes predictions and compute loss
            batch_score, _ = self._model(
                query_input_ids=train_batch['query_input_ids'].to(self._device),
                query_attention_mask=train_batch['query_attention_mask'].to(self._device),
                query_token_type_ids=train_batch['query_token_type_ids'].to(self._device),
                doc_text_emb=train_batch['doc_text_emb'].to(self._device),
                doc_entity_emb=train_batch['doc_entity_emb'].to(self._device),
            )

            batch_loss = self._criterion(batch_score, train_batch['label'].float().to(self._device))

            # Computes gradients
            batch_loss.backward()

            # Updates parameters
            self._optimizer.step()
            self._scheduler.step()

            # Returns the loss
            return batch_loss.item()

        # Returns the function that will be called inside the train loop
        return train_step

    def train(self):
        train_step = self.make_train_step()
        epoch_loss = 0
        num_batch = len(self._data_loader)
        for _, batch in tqdm.tqdm(enumerate(self._data_loader), total=num_batch):
            batch_loss = train_step(batch)
            epoch_loss += batch_loss

        return epoch_loss
