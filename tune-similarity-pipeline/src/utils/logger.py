tune-similarity-pipeline
├── src
│   ├── data
│   │   ├── preprocessing.py
│   │   ├── tokenizer.py
│   │   └── dataset.py
│   ├── models
│   │   ├── embedder.py
│   │   └── losses.py
│   ├── training
│   │   ├── trainer.py
│   │   └── callbacks.py
│   ├── evaluation
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── utils
│       ├── config.py
│       └── logger.py
├── scripts
│   ├── preprocess_data.py
│   └── train_model.py
├── configs
│   └── config.yaml
├── tests
│   └── test_preprocessing.py
├── requirements.txt
└── README.md