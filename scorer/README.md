# scorer

Scorer data and training (the MLLM stays frozen).

- `build_boxes.py`: grounds TextVQA answers on TextOCR word boxes.
- `build_selector_data.py`: merges those records with Visual-CoT evidence boxes. Each source keeps an image-disjoint held-out split, and every TextVQA val image is excluded.
- `train_selector.py`: trains the scorer with the listwise box-coverage loss and reports answer-token recall at the 5% budget.
