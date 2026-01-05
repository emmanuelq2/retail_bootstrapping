"""
Bootstrap script for retail sector intent classification dataset.
Loads synthetic data, performs data augmentation, and prepares training sets.
Includes synthetic data generation utilities using Faker and LLM prompts.
"""

import json
import random
import os
from datetime import datetime
from typing import List, Dict, Tuple
import re


# ============================================================================
# SYNTHETIC DATA GENERATION UTILITIES
# ============================================================================

def generate_synthetic_product():
    """
    Generate synthetic product data using Faker library.
    
    Requires: pip install faker
    
    Returns:
        dict: Synthetic product with realistic attributes
    """
    try:
        from faker import Faker
        fake = Faker()
        
        return {
            "product_id": fake.uuid4(),
            "name": fake.catch_phrase(),
            "category": random.choice(["Electronics", "Clothing", "Home", "Books", "Sports"]),
            "price": round(random.uniform(10, 500), 2),
            "rating": round(random.uniform(1, 5), 1),
            "reviews_count": random.randint(0, 10000),
            "in_stock": random.choice([True, False]),
            "description": fake.text(max_nb_chars=200)
        }
    except ImportError:
        print("Warning: Faker library not installed. Run: pip install faker")
        return None


def generate_with_openai_llm(prompt, n_samples=5, api_key=None):
    """
    Generate synthetic data using OpenAI's GPT models.
    
    Requires: pip install openai
    
    Args:
        prompt: LLM prompt for data generation
        n_samples: Number of samples to generate
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
    
    Returns:
        list: Generated synthetic data samples
    """
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=api_key) if api_key else OpenAI()
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data generator. Output only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            n=n_samples
        )
        return [json.loads(choice.message.content) for choice in response.choices]
    except ImportError:
        print("Warning: OpenAI library not installed. Run: pip install openai")
        return None
    except Exception as e:
        print(f"Error generating with OpenAI: {e}")
        return None


def generate_with_anthropic_claude(system_prompt, user_prompts, api_key=None):
    """
    Generate synthetic data using Anthropic's Claude models.
    
    Requires: pip install anthropic
    
    Args:
        system_prompt: System instructions for the model
        user_prompts: List of user prompts
        api_key: Anthropic API key (or set ANTHROPIC_API_KEY env var)
    
    Returns:
        list: Generated synthetic data samples
    """
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        results = []
        
        for user_prompt in user_prompts:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            results.append(json.loads(message.content[0].text))
        
        return results
    except ImportError:
        print("Warning: Anthropic library not installed. Run: pip install anthropic")
        return None
    except Exception as e:
        print(f"Error generating with Claude: {e}")
        return None


def create_llm_prompt_template(domain, fields, constraints=None):
    """
    Create structured prompts for LLM-based synthetic data generation.
    
    Args:
        domain: Domain description (e.g., "customer profile", "product review")
        fields: Dictionary of field names and their descriptions/types
        constraints: List of constraints or rules for data generation
    
    Returns:
        str: Formatted prompt for LLM
    """
    prompt = f"Generate realistic {domain} data with the following structure:\n\n"
    
    for field, field_type in fields.items():
        prompt += f"- {field}: {field_type}\n"
    
    if constraints:
        prompt += f"\nConstraints:\n"
        for constraint in constraints:
            prompt += f"- {constraint}\n"
    
    prompt += "\nReturn only valid JSON format."
    return prompt


def create_few_shot_prompt(examples, new_context):
    """
    Create few-shot learning prompt with examples.
    
    Args:
        examples: List of example data dictionaries
        new_context: Description of new data to generate
    
    Returns:
        str: Few-shot prompt for LLM
    """
    prompt = "Generate data following these examples:\n\n"
    
    for i, example in enumerate(examples, 1):
        prompt += f"Example {i}:\n{json.dumps(example, indent=2)}\n\n"
    
    prompt += f"Now generate similar data for:\n{new_context}\n\nReturn only JSON."
    return prompt


def augment_with_llm(original_data, variations=3, model="gpt-4"):
    """
    Generate variations of existing data using LLM.
    
    Args:
        original_data: Original data to augment
        variations: Number of variations to generate
        model: LLM model to use
    
    Returns:
        list: Augmented data variations
    """
    prompt = f"""
Original data:
{json.dumps(original_data, indent=2)}

Create {variations} realistic variations of this data while maintaining the same structure.
Change values realistically but keep the same schema.
Return as JSON array.
"""
    
    try:
        from openai import OpenAI
        client = OpenAI()
        
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in LLM augmentation: {e}")
        return None


# LLM PROMPT TEMPLATES FOR RETAIL DATA
# ============================================================================

RETAIL_PRODUCT_REVIEW_PROMPT = """
Generate a realistic product review with these fields:
- reviewer_name: string (realistic name)
- rating: integer (1-5)
- title: string (brief review title, 3-8 words)
- review_text: string (detailed review, 50-150 words)
- verified_purchase: boolean
- helpful_votes: integer (0-100)
- review_date: ISO date string (within last year)

Make the review authentic with varied sentiment based on rating.
Return only valid JSON.
"""

RETAIL_CUSTOMER_PROMPT = """
Generate a realistic customer profile with these fields:
- customer_id: unique identifier
- name: full name
- email: valid email format
- purchase_history: list of 1-10 product IDs
- total_spent: float (100-10000)
- loyalty_tier: "bronze" | "silver" | "gold" | "platinum"
- account_created: ISO date string
- last_purchase_date: ISO date string

Constraints:
- Total spent should correlate with loyalty tier
- Last purchase date should be after account creation
- Higher tiers should have longer purchase histories

Return only valid JSON.
"""

RETAIL_SUPPORT_TICKET_PROMPT = """
Generate a realistic customer support ticket:
- ticket_id: unique identifier
- customer_name: string
- issue_category: "delivery" | "refund" | "product_defect" | "account" | "other"
- priority: "low" | "medium" | "high" | "urgent"
- subject: string (concise issue description)
- description: string (detailed explanation, 50-200 words)
- status: "open" | "in_progress" | "resolved" | "closed"
- created_date: ISO date string
- tags: list of 1-3 relevant tags

Make it realistic with appropriate priority based on issue category.
Return only valid JSON.
"""

DATA_VALIDATION_PROMPT = """
Review this synthetic data and ensure:
1. All required fields are present
2. Data types are correct
3. Values are realistic and consistent
4. No duplicate IDs
5. Dates are valid and logical
6. Numerical ranges are appropriate
7. Text fields are meaningful

Data to validate:
{data}

Return: {{"valid": true/false, "issues": [list of specific issues if any]}}
"""


class RetailDatasetBootstrap:
    """Bootstrap synthetic retail dataset for intent classification."""
    
    def __init__(self, dataset_path: str = "retail_dataset.json"):
        self.dataset_path = dataset_path
        self.data = None
        self.training_data = []
        self.validation_data = []
        self.test_data = []
        
    def load_dataset(self) -> None:
        """Load the synthetic dataset from JSON file."""
        print(f"Loading dataset from {self.dataset_path}...")
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        total_examples = sum(len(intent['examples']) for intent in self.data['intents'])
        print(f" Loaded {len(self.data['intents'])} intents with {total_examples} examples")
        
    def augment_data(self, augmentation_factor: float = 1.5) -> None:
        """
        Augment dataset with variations of existing examples.
        
        Args:
            augmentation_factor: Multiplier for dataset size (1.5 = 50% more data)
        """
        print("\nAugmenting dataset...")
        
        augmentation_templates = {
            'prefix': [
                "I would like to ",
                "Please help me ",
                "Could you ",
                "I'm trying to ",
                "I want to ",
                "Can you help me ",
            ],
            'suffix': [
                " please",
                " right now",
                " as soon as possible",
                " today",
                " immediately",
            ],
            'replacements': {
                'buy': ['purchase', 'get', 'order', 'acquire'],
                'need': ['require', 'want', 'would like'],
                'help': ['assist', 'support', 'aid'],
                'order': ['purchase', 'buy', 'get'],
            }
        }
        
        for intent in self.data['intents']:
            original_count = len(intent['examples'])
            target_count = int(original_count * augmentation_factor)
            augmented = list(intent['examples'])
            
            while len(augmented) < target_count:
                # Pick a random original example
                base_example = random.choice(intent['examples'])
                
                # Handle both dict and string examples
                if isinstance(base_example, dict):
                    base_text = base_example['text']
                    base_slots = base_example.get('slots', [])
                else:
                    base_text = base_example
                    base_slots = []
                
                # Apply random augmentation
                aug_type = random.choice(['prefix', 'suffix', 'replacement', 'case'])
                
                if aug_type == 'prefix':
                    new_text = random.choice(augmentation_templates['prefix']) + base_text.lower()
                elif aug_type == 'suffix':
                    new_text = base_text.rstrip('?.!') + random.choice(augmentation_templates['suffix'])
                elif aug_type == 'replacement':
                    new_text = base_text
                    for word, replacements in augmentation_templates['replacements'].items():
                        if word in new_text.lower():
                            new_text = re.sub(
                                r'\b' + word + r'\b',
                                random.choice(replacements),
                                new_text,
                                flags=re.IGNORECASE
                            )
                            break
                else:  # case variation
                    new_text = base_text.lower() if base_text[0].isupper() else base_text.capitalize()
                
                # Create new example with slots
                if isinstance(base_example, dict):
                    new_example = {'text': new_text, 'slots': base_slots}
                    # Avoid duplicates by checking text
                    if not any(e.get('text') == new_text if isinstance(e, dict) else e == new_text for e in augmented):
                        augmented.append(new_example)
                else:
                    if new_text not in augmented:
                        augmented.append(new_text)
            
            intent['examples'] = augmented
            print(f" {intent['intent']}: {original_count} -> {len(augmented)} examples")
    
    def split_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> None:
        """
        Split dataset into training, validation, and test sets.
        
        Args:
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set (remainder goes to test)
        """
        print("\nSplitting dataset...")
        
        for intent in self.data['intents']:
            examples = intent['examples']
            random.shuffle(examples)
            
            n_total = len(examples)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            train_examples = examples[:n_train]
            val_examples = examples[n_train:n_train + n_val]
            test_examples = examples[n_train + n_val:]
            
            # Add to respective datasets
            for example in train_examples:
                if isinstance(example, dict):
                    self.training_data.append({
                        "text": example['text'],
                        "intent": intent['intent'],
                        "slots": example.get('slots', [])
                    })
                else:
                    self.training_data.append({"text": example, "intent": intent['intent'], "slots": []})
                    
            for example in val_examples:
                if isinstance(example, dict):
                    self.validation_data.append({
                        "text": example['text'],
                        "intent": intent['intent'],
                        "slots": example.get('slots', [])
                    })
                else:
                    self.validation_data.append({"text": example, "intent": intent['intent'], "slots": []})
                    
            for example in test_examples:
                if isinstance(example, dict):
                    self.test_data.append({
                        "text": example['text'],
                        "intent": intent['intent'],
                        "slots": example.get('slots', [])
                    })
                else:
                    self.test_data.append({"text": example, "intent": intent['intent'], "slots": []})
        
        # Shuffle the splits
        random.shuffle(self.training_data)
        random.shuffle(self.validation_data)
        random.shuffle(self.test_data)
        
        print(f"  [OK] Training: {len(self.training_data)} examples")
        print(f"  [OK] Validation: {len(self.validation_data)} examples")
        print(f"  [OK] Test: {len(self.test_data)} examples")
    
    def save_splits(self, output_dir: str = "data") -> None:
        """Save the split datasets to separate files."""
        print(f"\nSaving split datasets to '{output_dir}' directory...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training data
        with open(os.path.join(output_dir, "train.json"), 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, indent=2, ensure_ascii=False)
        
        # Save validation data
        with open(os.path.join(output_dir, "validation.json"), 'w', encoding='utf-8') as f:
            json.dump(self.validation_data, f, indent=2, ensure_ascii=False)
        
        # Save test data
        with open(os.path.join(output_dir, "test.json"), 'w', encoding='utf-8') as f:
            json.dump(self.test_data, f, indent=2, ensure_ascii=False)
        
        print(f"  [OK] Saved train.json ({len(self.training_data)} examples)")
        print(f"  [OK] Saved validation.json ({len(self.validation_data)} examples)")
        print(f"  [OK] Saved test.json ({len(self.test_data)} examples)")
    
    def generate_statistics(self) -> Dict:
        """Generate statistics about the dataset."""
        stats = {
            "total_examples": len(self.training_data) + len(self.validation_data) + len(self.test_data),
            "training_examples": len(self.training_data),
            "validation_examples": len(self.validation_data),
            "test_examples": len(self.test_data),
            "intents": {},
            "slots": {},
            "generated_date": datetime.now().isoformat()
        }
        
        # Count examples per intent and slots
        for item in self.training_data + self.validation_data + self.test_data:
            intent = item['intent']
            if intent not in stats['intents']:
                stats['intents'][intent] = 0
            stats['intents'][intent] += 1
            
            # Count slots
            for slot in item.get('slots', []):
                slot_type = slot.get('type')
                if slot_type:
                    if slot_type not in stats['slots']:
                        stats['slots'][slot_type] = 0
                    stats['slots'][slot_type] += 1
        
        return stats
    
    def save_statistics(self, output_dir: str = "data") -> None:
        """Save dataset statistics."""
        stats = self.generate_statistics()
        
        with open(os.path.join(output_dir, "statistics.json"), 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("Dataset Statistics:")
        print(f"{'='*60}")
        print(f"Total examples: {stats['total_examples']}")
        print(f"Training: {stats['training_examples']}")
        print(f"Validation: {stats['validation_examples']}")
        print(f"Test: {stats['test_examples']}")
        print(f"\nIntent distribution:")
        for intent, count in stats['intents'].items():
            print(f"  - {intent}: {count} examples")
        
        if stats.get('slots'):
            print(f"\nSlot types:")
            for slot_type, count in stats['slots'].items():
                print(f"  - {slot_type}: {count} occurrences")
        
        print(f"{'='*60}")
    
    def run(self, augment: bool = True, augmentation_factor: float = 1.5) -> None:
        """
        Run the complete bootstrap process.
        
        Args:
            augment: Whether to perform data augmentation
            augmentation_factor: Multiplier for augmentation
        """
        print("="*60)
        print("RETAIL SECTOR DATASET BOOTSTRAP")
        print("="*60)
        
        self.load_dataset()
        
        if augment:
            self.augment_data(augmentation_factor)
        
        self.split_dataset()
        self.save_splits()
        self.save_statistics()
        
        print("\n[OK] Bootstrap completed successfully!")


def main():
    """Main entry point for the bootstrap script."""
    random.seed(42)  # For reproducibility
    
    bootstrap = RetailDatasetBootstrap("retail_dataset.json")
    bootstrap.run(augment=True, augmentation_factor=1.5)


if __name__ == "__main__":
    main()
