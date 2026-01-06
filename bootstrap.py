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


# ============================================================================
# APPROACH #1: MODEL-ASSISTED SYNTHETIC GENERATION
# ============================================================================
# Uses LLMs (GPT, Claude) to generate realistic synthetic data based on prompts.
# Benefits: High diversity, natural language quality, contextual understanding
# Trade-offs: Requires API keys, costs money, slower than rule-based methods
# Use when: Need high-quality diverse data, have budget for API calls
# ============================================================================

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


# ============================================================================
# APPROACH #2: CONTROLLED SYNTHETIC GENERATION
# ============================================================================
# Creates structured prompts with explicit schemas, constraints, and rules.
# Benefits: Predictable output format, schema compliance, constraint satisfaction
# Trade-offs: Less creative than free-form generation, requires prompt engineering
# Use when: Need consistent data format, strict validation requirements
# ============================================================================

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


# ============================================================================
# APPROACH #3: HUMAN-IN-THE-LOOP (HITL) SYNTHETIC GENERATION
# ============================================================================
# Combines automated generation with human review, correction, and validation.
# Benefits: High quality, catches edge cases, domain expertise integration
# Trade-offs: Time-intensive, requires human resources, not fully automated
# Use when: Quality > speed, domain complexity requires human judgment
# ============================================================================

def create_hitl_review_batch(synthetic_samples, batch_size=10):
    """
    Prepare synthetic data samples for human review in manageable batches.
    
    HITL Workflow:
    1. Generate synthetic samples (automated)
    2. Present to human reviewers in batches
    3. Collect feedback (correct/incorrect, suggestions)
    4. Update generation rules based on feedback
    5. Iterate until quality threshold met
    
    Args:
        synthetic_samples: List of generated samples
        batch_size: Number of samples per review batch
    
    Returns:
        list: Batched samples ready for review
    """
    batches = []
    for i in range(0, len(synthetic_samples), batch_size):
        batch = {
            "batch_id": i // batch_size + 1,
            "samples": synthetic_samples[i:i + batch_size],
            "review_status": "pending",
            "reviewer_notes": [],
            "corrections_needed": []
        }
        batches.append(batch)
    return batches


def apply_human_corrections(sample, corrections):
    """
    Apply human corrections to a synthetic sample.
    
    Args:
        sample: Original synthetic sample
        corrections: Dict with field-level corrections
            Example: {"text": "corrected text", "intent": "correct_intent"}
    
    Returns:
        dict: Corrected sample with metadata
    """
    corrected_sample = sample.copy()
    corrected_sample.update(corrections)
    corrected_sample["metadata"] = {
        "human_reviewed": True,
        "original_sample": sample,
        "corrections_applied": list(corrections.keys()),
        "review_date": datetime.now().isoformat()
    }
    return corrected_sample


def extract_feedback_patterns(reviewed_batches):
    """
    Analyze human feedback to identify common correction patterns.
    
    This enables learning from human reviewers to improve automated generation.
    
    Args:
        reviewed_batches: List of review batches with corrections
    
    Returns:
        dict: Common patterns and suggested improvements
    """
    patterns = {
        "common_errors": {},
        "correction_frequency": {},
        "quality_score": 0.0,
        "suggested_rules": []
    }
    
    total_samples = 0
    total_corrections = 0
    
    for batch in reviewed_batches:
        for sample in batch.get("samples", []):
            total_samples += 1
            if sample.get("metadata", {}).get("corrections_applied"):
                total_corrections += len(sample["metadata"]["corrections_applied"])
                
                # Track which fields need correction most often
                for field in sample["metadata"]["corrections_applied"]:
                    patterns["correction_frequency"][field] = \
                        patterns["correction_frequency"].get(field, 0) + 1
    
    # Calculate quality score (% of samples needing no corrections)
    if total_samples > 0:
        patterns["quality_score"] = 1 - (total_corrections / total_samples)
    
    return patterns


# ============================================================================
# APPROACH #4: ACTIVE LEARNING & UNCERTAINTY-DRIVEN SAMPLING
# ============================================================================
# Selectively generate samples where model is most uncertain or performance is weakest.
# Benefits: Efficient use of labeling resources, targets model weaknesses
# Trade-offs: Requires trained model, iterative process, computational overhead
# Use when: Limited labeling budget, want to maximize model performance gains
# ============================================================================

def select_uncertain_samples(model, unlabeled_pool, n_samples=100, method='entropy'):
    """
    Select samples where the model is most uncertain for labeling/generation.
    
    Active Learning Strategies:
    - 'entropy': Maximum prediction entropy (most confused)
    - 'margin': Smallest margin between top 2 predictions
    - 'least_confident': Lowest confidence in top prediction
    - 'disagreement': Maximum disagreement in ensemble
    
    Args:
        model: Trained classifier (must have predict_proba method)
        unlabeled_pool: List of unlabeled samples
        n_samples: Number of samples to select
        method: Uncertainty sampling strategy
    
    Returns:
        list: Most uncertain samples for human labeling or synthetic generation
    """
    # DRAFT IMPLEMENTATION - Requires actual model
    # This is a skeleton showing the approach
    
    uncertainties = []
    
    for sample in unlabeled_pool:
        try:
            # Get model predictions (probabilities for each class)
            probs = model.predict_proba([sample['text']])[0]
            
            if method == 'entropy':
                # Calculate Shannon entropy: -sum(p * log(p))
                import numpy as np
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                uncertainty = entropy
                
            elif method == 'margin':
                # Margin between top 2 predictions
                sorted_probs = sorted(probs, reverse=True)
                margin = sorted_probs[0] - sorted_probs[1]
                uncertainty = -margin  # Lower margin = higher uncertainty
                
            elif method == 'least_confident':
                # 1 - max(probabilities)
                uncertainty = 1 - max(probs)
            
            uncertainties.append((sample, uncertainty))
            
        except Exception as e:
            print(f"Error calculating uncertainty: {e}")
            continue
    
    # Sort by uncertainty (descending) and return top n
    uncertainties.sort(key=lambda x: x[1], reverse=True)
    return [sample for sample, _ in uncertainties[:n_samples]]


def generate_similar_to_hard_examples(hard_examples, llm_client, n_variations=5):
    """
    Generate synthetic variations of examples where model struggles.
    
    Args:
        hard_examples: Samples with high uncertainty or low model performance
        llm_client: LLM client for generation
        n_variations: Number of variations per hard example
    
    Returns:
        list: Generated synthetic samples targeting model weaknesses
    """
    synthetic_variants = []
    
    for example in hard_examples:
        prompt = f"""
Generate {n_variations} variations of this challenging example.
Keep the same intent but vary the phrasing significantly.

Original: {example['text']}
Intent: {example.get('intent', 'unknown')}

Return as JSON array: [{{"text": "...", "intent": "..."}}]
"""
        # Call LLM to generate variations
        # synthetic_variants.extend(llm_client.generate(prompt))
        
    return synthetic_variants


# ============================================================================
# APPROACH #5: CONTRASTIVE & ADVERSARIAL GENERATION
# ============================================================================
# Generate hard negatives and near-misses to improve model discrimination.
# Benefits: Improves model robustness, better decision boundaries
# Trade-offs: Complex generation logic, requires careful design
# Use when: Model confuses similar classes, need better class separation
# ============================================================================

def generate_hard_negatives(positive_example, target_intent, confusable_intents):
    """
    Generate adversarial examples that look similar but have different intent.
    
    Hard Negative Types:
    1. MINIMAL EDITS: Change 1-2 words to flip intent
       - "I want to buy" → "I don't want to buy" (negation)
    2. CONFUSABLE PHRASES: Use phrases from similar intents
       - "track order" vs "cancel order" (same domain, different action)
    3. AMBIGUOUS CONTEXT: Remove disambiguating information
       - "I need help with my account password" → "I need help" (underspecified)
    
    Args:
        positive_example: Original correctly classified example
        target_intent: The intent we want to discriminate from
        confusable_intents: List of intents commonly confused with target
    
    Returns:
        list: Hard negative examples
    """
    hard_negatives = []
    
    # STRATEGY 1: Negation injection
    negation_patterns = [
        (r'\bwant to\b', "don't want to"),
        (r'\bneed\b', "don't need"),
        (r'\bcan you\b', "can you not"),
        (r'\bhelp me\b', "stop helping me"),
    ]
    
    for pattern, replacement in negation_patterns:
        negative_text = re.sub(pattern, replacement, positive_example['text'], flags=re.IGNORECASE)
        if negative_text != positive_example['text']:
            hard_negatives.append({
                "text": negative_text,
                "intent": f"NOT_{target_intent}",
                "generation_strategy": "negation",
                "original_intent": target_intent
            })
    
    # STRATEGY 2: Action substitution (for confusable intents)
    action_swaps = {
        "buy": ["return", "cancel", "track"],
        "track": ["cancel", "buy", "return"],
        "cancel": ["track", "buy", "modify"],
    }
    
    # STRATEGY 3: Context removal (create ambiguity)
    # Remove specific details that disambiguate intent
    
    return hard_negatives


def generate_contrastive_pairs(dataset, similarity_threshold=0.8):
    """
    Create positive/negative pairs for contrastive learning.
    
    Contrastive Pairs:
    - POSITIVE: Same intent, different phrasing
    - NEGATIVE: Different intent, similar phrasing (hard negatives)
    
    Args:
        dataset: List of labeled examples
        similarity_threshold: Minimum similarity for hard negatives
    
    Returns:
        list: Tuples of (anchor, positive, negative)
    """
    contrastive_triplets = []
    
    # Group by intent
    intent_groups = {}
    for sample in dataset:
        intent = sample.get('intent')
        if intent not in intent_groups:
            intent_groups[intent] = []
        intent_groups[intent].append(sample)
    
    # Generate triplets
    for intent, samples in intent_groups.items():
        for anchor in samples:
            # Positive: Random sample from same intent
            positive = random.choice([s for s in samples if s != anchor])
            
            # Negative: Sample from different intent (preferably similar text)
            negative_candidates = [s for s in dataset if s.get('intent') != intent]
            negative = random.choice(negative_candidates) if negative_candidates else None
            
            if negative:
                contrastive_triplets.append({
                    "anchor": anchor,
                    "positive": positive,
                    "negative": negative,
                    "intent": intent
                })
    
    return contrastive_triplets


# ============================================================================
# APPROACH #7: MULTIMODAL & CROSS-MODAL SYNTHETIC DATA
# ============================================================================
# Generate synthetic data across multiple modalities (text, image, audio).
# Benefits: Richer representations, cross-modal learning, real-world applicability
# Trade-offs: Complex pipeline, requires multimodal models, higher compute
# Use when: Building multimodal applications, have multimodal models available
# ============================================================================

def generate_text_from_image_description(image_description, llm_client):
    """
    Generate text utterances based on product image descriptions.
    
    Use Case: E-commerce where users describe what they see
    Example: Image shows "red sneakers" → "I'm looking for those red shoes"
    
    Args:
        image_description: Description of image content
        llm_client: LLM for text generation
    
    Returns:
        list: Text utterances related to image content
    """
    prompt = f"""
Generate 5 natural customer queries someone might say when looking at this product:

Product visual description: {image_description}

Make queries conversational and varied (questions, statements, comparisons).
Return as JSON array.
"""
    # return llm_client.generate(prompt)
    return []  # DRAFT - requires LLM client


def generate_audio_transcription_variants(base_transcript, error_rate=0.1):
    """
    Simulate ASR (speech recognition) errors in transcripts.
    
    Common ASR errors:
    - Homophones: "buy" → "by", "two" → "to"
    - Missing words: "I want to buy" → "I want buy"
    - Extra filler words: "I want" → "um I want like"
    
    Args:
        base_transcript: Clean text transcript
        error_rate: Probability of introducing errors (0-1)
    
    Returns:
        list: Noisy transcript variations
    """
    homophones = {
        "buy": ["by", "bye"],
        "to": ["two", "too"],
        "their": ["there", "they're"],
        "order": ["border"],
    }
    
    fillers = ["um", "uh", "like", "you know", "so"]
    
    variants = []
    words = base_transcript.split()
    
    # Generate noisy variant
    noisy_words = []
    for word in words:
        if random.random() < error_rate:
            # Apply random error type
            if word.lower() in homophones and random.random() < 0.5:
                # Homophone substitution
                noisy_words.append(random.choice(homophones[word.lower()]))
            elif random.random() < 0.3:
                # Insert filler word
                noisy_words.append(random.choice(fillers))
                noisy_words.append(word)
            # else: word deletion (skip word)
        else:
            noisy_words.append(word)
    
    variants.append(" ".join(noisy_words))
    return variants


# ============================================================================
# APPROACH #8: PROGRAMMATIC DATA GENERATION WITH CONSTRAINT SOLVERS
# ============================================================================
# Use formal methods (SMT/SAT solvers) to generate data satisfying constraints.
# Benefits: Guaranteed constraint satisfaction, systematic coverage
# Trade-offs: Complex setup, limited to structured domains, performance overhead
# Use when: Need provably correct data, formal specifications available
# ============================================================================

def generate_with_constraints(schema, constraints):
    """
    Generate synthetic data satisfying formal constraints using SMT solvers.
    
    Example Constraints:
    - total_price = sum(item_prices)
    - delivery_date > order_date
    - loyalty_tier = "gold" IFF total_spent > 5000
    - all(product_ids) are unique
    
    Requires: pip install z3-solver
    
    Args:
        schema: Data schema definition
        constraints: List of constraint expressions
    
    Returns:
        dict: Generated data satisfying all constraints
    """
    # DRAFT IMPLEMENTATION - Requires Z3 solver
    try:
        from z3 import *
        
        # Example: Generate shopping cart with constraints
        solver = Solver()
        
        # Define variables
        # item_price_1 = Real('item_price_1')
        # item_price_2 = Real('item_price_2')
        # total_price = Real('total_price')
        
        # Add constraints
        # solver.add(total_price == item_price_1 + item_price_2)
        # solver.add(item_price_1 > 0, item_price_1 < 1000)
        # solver.add(item_price_2 > 0, item_price_2 < 1000)
        # solver.add(total_price < 1500)
        
        # if solver.check() == sat:
        #     model = solver.model()
        #     return {var: model[var] for var in model}
        
        return {}
        
    except ImportError:
        print("Warning: Z3 solver not installed. Run: pip install z3-solver")
        return None


def validate_constraints(sample, constraints):
    """
    Validate that generated sample satisfies all constraints.
    
    Args:
        sample: Generated data sample
        constraints: List of constraint functions
    
    Returns:
        tuple: (is_valid, violations)
    """
    violations = []
    
    for constraint in constraints:
        try:
            if not constraint(sample):
                violations.append(constraint.__name__)
        except Exception as e:
            violations.append(f"{constraint.__name__}: {str(e)}")
    
    return len(violations) == 0, violations


# ============================================================================
# APPROACH #9: REINFORCEMENT LEARNING & ITERATIVE FEEDBACK
# ============================================================================
# Use RL to learn optimal data generation policies based on model performance.
# Benefits: Automatically discovers effective generation strategies
# Trade-offs: Complex training, requires reward signal, computationally expensive
# Use when: Have clear performance metrics, can afford iterative training
# ============================================================================

def train_generator_with_rl(generator, discriminator, reward_fn, n_iterations=1000):
    """
    Train data generator using reinforcement learning with model feedback.
    
    RL Framework:
    - AGENT: Data generator (policy network)
    - ACTION: Generate synthetic sample
    - STATE: Current dataset statistics
    - REWARD: Model performance improvement, data quality score
    - ENVIRONMENT: Training/validation loop
    
    Args:
        generator: Generator model/function
        discriminator: Model being trained on synthetic data
        reward_fn: Function computing reward based on model performance
        n_iterations: Training iterations
    
    Returns:
        trained generator
    """
    # DRAFT - Pseudo-code for RL-based generation
    
    for iteration in range(n_iterations):
        # 1. Generate synthetic batch
        # synthetic_batch = generator.generate()
        
        # 2. Train discriminator on synthetic data
        # discriminator.train(synthetic_batch)
        
        # 3. Evaluate discriminator performance
        # performance = discriminator.evaluate(validation_set)
        
        # 4. Compute reward (e.g., accuracy improvement)
        # reward = reward_fn(performance)
        
        # 5. Update generator policy using reward
        # generator.update_policy(reward)
        
        pass
    
    return generator


def iterative_refinement_loop(initial_data, model_class, n_rounds=5):
    """
    Iteratively refine synthetic data based on model performance.
    
    Refinement Loop:
    1. Generate initial synthetic data
    2. Train model on synthetic data
    3. Identify failure cases / low-confidence predictions
    4. Generate more data targeting weaknesses
    5. Repeat until performance plateau
    
    Args:
        initial_data: Starting synthetic dataset
        model_class: Model class to train
        n_rounds: Number of refinement iterations
    
    Returns:
        tuple: (refined_data, trained_model, performance_history)
    """
    current_data = initial_data
    performance_history = []
    
    for round_num in range(n_rounds):
        print(f"\n=== Refinement Round {round_num + 1} ===")
        
        # Train model on current data
        # model = model_class()
        # model.train(current_data)
        
        # Evaluate performance
        # performance = model.evaluate(validation_set)
        # performance_history.append(performance)
        
        # Identify weaknesses
        # weak_examples = identify_failure_cases(model, validation_set)
        
        # Generate targeted synthetic data
        # new_synthetic = generate_similar_to_hard_examples(weak_examples)
        
        # Augment dataset
        # current_data.extend(new_synthetic)
        
        pass
    
    return current_data, None, performance_history


# ============================================================================
# APPROACH #10: EVALUATION-DRIVEN DATA CREATION
# ============================================================================
# Generate data specifically designed to test model capabilities and edge cases.
# Benefits: Comprehensive evaluation, identifies blind spots, test coverage
# Trade-offs: Requires expertise to design tests, may not reflect real distribution
# Use when: Building test suites, validating model robustness, compliance testing
# ============================================================================

def generate_edge_case_tests(intent_schema):
    """
    Generate edge cases and boundary conditions for comprehensive testing.
    
    Edge Case Categories:
    1. MINIMAL INPUT: Single word, empty string
    2. MAXIMAL INPUT: Very long text (> 500 words)
    3. SPECIAL CHARACTERS: Unicode, emojis, symbols
    4. AMBIGUOUS: Could match multiple intents
    5. OUT-OF-DISTRIBUTION: Novel combinations not in training
    6. ADVERSARIAL: Intentionally confusing
    
    Args:
        intent_schema: Schema defining intents and expected inputs
    
    Returns:
        list: Edge case test samples
    """
    edge_cases = []
    
    for intent in intent_schema['intents']:
        # EDGE CASE 1: Minimal input
        edge_cases.append({
            "text": intent['name'].split('_')[0],  # Single word
            "intent": intent['name'],
            "test_type": "minimal_input",
            "expected_behavior": "should_classify" if intent.get('allow_minimal') else "should_reject"
        })
        
        # EDGE CASE 2: Maximum length
        long_text = " ".join([intent.get('example', 'test')] * 100)
        edge_cases.append({
            "text": long_text,
            "intent": intent['name'],
            "test_type": "maximal_input"
        })
        
        # EDGE CASE 3: Special characters
        edge_cases.append({
            "text": f"🛒 {intent.get('example', 'test')} 💳 !!!",
            "intent": intent['name'],
            "test_type": "special_characters"
        })
        
        # EDGE CASE 4: Ambiguous (multiple intents possible)
        # Would require cross-intent analysis
        
    return edge_cases


def generate_robustness_tests(base_examples, perturbation_types=None):
    """
    Generate robustness tests by perturbing valid examples.
    
    Perturbation Types:
    - TYPOS: "buy product" → "byy prodcut"
    - WORD_SWAP: "I want to buy" → "I buy to want"
    - SYNONYM: "purchase" → "acquire"
    - PARAPHRASE: Complete rephrase with same meaning
    - STYLE_CHANGE: Formal ↔ Casual
    
    Args:
        base_examples: Original valid examples
        perturbation_types: List of perturbation strategies to apply
    
    Returns:
        list: Robustness test cases with expected behavior
    """
    if perturbation_types is None:
        perturbation_types = ['typo', 'word_swap', 'synonym']
    
    robustness_tests = []
    
    for example in base_examples:
        original_text = example['text']
        words = original_text.split()
        
        # TYPO perturbation
        if 'typo' in perturbation_types and len(words) > 0:
            typo_text = original_text
            # Randomly swap 2 adjacent characters in a word
            target_word = random.choice(words)
            if len(target_word) > 2:
                idx = random.randint(0, len(target_word) - 2)
                typo_word = target_word[:idx] + target_word[idx+1] + target_word[idx] + target_word[idx+2:]
                typo_text = original_text.replace(target_word, typo_word, 1)
            
            robustness_tests.append({
                "text": typo_text,
                "intent": example['intent'],
                "perturbation": "typo",
                "original": original_text,
                "expected_behavior": "should_classify_correctly"
            })
        
        # WORD SWAP perturbation
        if 'word_swap' in perturbation_types and len(words) > 2:
            swapped_words = words.copy()
            i, j = random.sample(range(len(words)), 2)
            swapped_words[i], swapped_words[j] = swapped_words[j], swapped_words[i]
            
            robustness_tests.append({
                "text": " ".join(swapped_words),
                "intent": example['intent'],
                "perturbation": "word_swap",
                "original": original_text,
                "expected_behavior": "may_fail"  # Depends on semantic preservation
            })
    
    return robustness_tests


def create_evaluation_suite(intents, coverage_criteria=None):
    """
    Create comprehensive evaluation suite ensuring test coverage.
    
    Coverage Criteria:
    - INTENT_COVERAGE: At least N examples per intent
    - SLOT_COVERAGE: All slot types represented
    - LENGTH_COVERAGE: Short, medium, long examples
    - STYLE_COVERAGE: Formal, casual, question, statement
    - EDGE_CASE_COVERAGE: Boundary conditions
    
    Args:
        intents: List of intent definitions
        coverage_criteria: Dict specifying coverage requirements
    
    Returns:
        dict: Evaluation suite with coverage report
    """
    if coverage_criteria is None:
        coverage_criteria = {
            "min_per_intent": 10,
            "length_bins": [5, 15, 50],  # words
            "styles": ["formal", "casual", "question"]
        }
    
    evaluation_suite = {
        "test_cases": [],
        "coverage_report": {
            "intent_coverage": {},
            "length_coverage": {},
            "style_coverage": {}
        }
    }
    
    # Generate tests ensuring coverage
    for intent in intents:
        intent_tests = []
        
        # Ensure minimum examples per intent
        for _ in range(coverage_criteria['min_per_intent']):
            # Generate test case
            # intent_tests.append(generate_test_for_intent(intent))
            pass
        
        evaluation_suite["test_cases"].extend(intent_tests)
    
    return evaluation_suite


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
        APPROACH #6: DATA AUGMENTATION PIPELINE (RULE-BASED)
        
        Augment dataset with variations of existing examples using deterministic rules.
        
        Techniques used:
        1. PREFIX INJECTION - Add polite/urgent phrases at the beginning
        2. SUFFIX INJECTION - Add temporal/polite modifiers at the end
        3. SYNONYM REPLACEMENT - Swap words with semantically similar alternatives
        4. CASE VARIATION - Toggle capitalization patterns
        
        Benefits: Fast, deterministic, no API costs, preserves intent/slots
        Trade-offs: Limited creativity, potential for unnatural combinations
        Use when: Need quick dataset expansion with controlled variations
        
        Args:
            augmentation_factor: Multiplier for dataset size (1.5 = 50% more data)
        """
        print("\nAugmenting dataset...")
        
        # RULE-BASED AUGMENTATION TEMPLATES
        # These templates create controlled variations while maintaining semantic meaning
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
        
        # Process each intent separately to maintain class balance
        for intent in self.data['intents']:
            original_count = len(intent['examples'])
            target_count = int(original_count * augmentation_factor)
            augmented = list(intent['examples'])
            
            # Generate variations until we reach target count
            while len(augmented) < target_count:
                # STEP 1: Sample a base example from original data (with replacement)
                base_example = random.choice(intent['examples'])
                
                # STEP 2: Extract text and slot information
                # Handle both dict format {"text": ..., "slots": ...} and plain strings
                if isinstance(base_example, dict):
                    base_text = base_example['text']
                    base_slots = base_example.get('slots', [])  # Preserve slot annotations
                else:
                    base_text = base_example
                    base_slots = []
                
                # STEP 3: Randomly select augmentation strategy
                # Equal probability for each transformation type
                aug_type = random.choice(['prefix', 'suffix', 'replacement', 'case'])
                
                # STEP 4: Apply selected augmentation strategy
                if aug_type == 'prefix':
                    # PREFIX INJECTION: Prepend polite/action phrases
                    # Example: "cancel order" -> "I would like to cancel order"
                    new_text = random.choice(augmentation_templates['prefix']) + base_text.lower()
                    
                elif aug_type == 'suffix':
                    # SUFFIX INJECTION: Append temporal/polite modifiers
                    # Example: "track my package" -> "track my package please"
                    new_text = base_text.rstrip('?.!') + random.choice(augmentation_templates['suffix'])
                    
                elif aug_type == 'replacement':
                    # SYNONYM REPLACEMENT: Swap keywords with synonyms
                    # Example: "I need help" -> "I require assistance"
                    new_text = base_text
                    for word, replacements in augmentation_templates['replacements'].items():
                        if word in new_text.lower():
                            # Use word boundary regex to avoid partial matches
                            new_text = re.sub(
                                r'\b' + word + r'\b',
                                random.choice(replacements),
                                new_text,
                                flags=re.IGNORECASE
                            )
                            break  # Only replace one word per augmentation
                            
                else:  # case variation
                    # CASE VARIATION: Toggle capitalization
                    # Example: "Help me" -> "help me" or vice versa
                    new_text = base_text.lower() if base_text[0].isupper() else base_text.capitalize()
                
                # STEP 5: Create augmented example and deduplicate
                # Preserve original data structure (dict or string)
                if isinstance(base_example, dict):
                    new_example = {'text': new_text, 'slots': base_slots}
                    # DEDUPLICATION: Avoid exact text matches to prevent data leakage
                    if not any(e.get('text') == new_text if isinstance(e, dict) else e == new_text for e in augmented):
                        augmented.append(new_example)
                else:
                    # Simple string format - direct deduplication
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
