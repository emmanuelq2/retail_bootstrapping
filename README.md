# Retail Sector Intent Classification Dataset

This project contains a synthetic dataset for retail sector intent classification with bootstrapping capabilities.

## Intents

The dataset includes the following intents:

1. **order_product** - Customer wants to purchase or order items
2. **contact_customer_service** - Customer wants to reach support or get assistance

## Dataset Structure

- `retail_dataset.json` - Original synthetic dataset with examples for each intent
- `bootstrap.py` - Script to augment, split, and prepare the dataset
- `data/` - Output directory containing processed datasets

## Usage

### Run Bootstrap

```bash
python bootstrap.py
```

This will:
1. Load the synthetic dataset
2. Augment the data (50% increase by default)
3. Split into train/validation/test sets (70/15/15)
4. Save processed datasets to `data/` directory
5. Generate statistics

### Output Files

After bootstrapping, you'll find:

- `data/train.json` - Training dataset
- `data/validation.json` - Validation dataset
- `data/test.json` - Test dataset
- `data/statistics.json` - Dataset statistics

## Dataset Features

- **Original Examples**: 60 (30 per intent)
- **Augmented Examples**: ~90 (with augmentation factor of 1.5)
- **Data Augmentation**: Includes prefix/suffix additions, word replacements, and case variations
- **Reproducible**: Uses fixed random seed for consistent splits

## Example Usage

```python
from bootstrap import RetailDatasetBootstrap

# Initialize bootstrap
bootstrap = RetailDatasetBootstrap("retail_dataset.json")

# Run with custom augmentation
bootstrap.run(augment=True, augmentation_factor=2.0)
```

## Requirements

- Python 3.6+
- Standard library only (no external dependencies)

## License

MIT License
