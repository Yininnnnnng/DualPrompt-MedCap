# -*- coding: utf-8 -*-
"""SLAKE evaluation matrics test
"""

!pip install open_clip_torch
!pip install einops_exts

!pip install spacy scispacy
!pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_lg-0.5.1.tar.gz

from google.colab import drive
import torch
import open_clip
from PIL import Image
import spacy
import re
import numpy as np
from scispacy.linking import EntityLinker
from tqdm import tqdm
import datetime
import traceback
import json
import io
import pandas as pd
import os
from IPython.display import display


drive.mount('/content/drive')

class MedicalImageCaptionEvaluator:
    def __init__(self):
        print("Initializing evaluator...")


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # 1. BiomedCLIP
        print("Loading BiomedCLIP...")
        self.model, self.preprocess = open_clip.create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.tokenizer = open_clip.get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.model.to(self.device)

        # 2. NLP
        print("Loading Medical NLP model...")
        self.nlp = spacy.load("en_core_sci_lg")
        self.nlp.add_pipe("scispacy_linker", config={"linker_name": "umls"})

        # 3. dictionary
        self.medical_patterns = {
            'modality_terms': [
                'CT scan', 'MRI', 'X-ray', 'radiograph', 'MRI scan',
                'chest X-ray', 'abdominal CT', 'ultrasound', 'PET scan'
            ],
            'anatomy_terms': [
                'brain', 'lung', 'heart', 'liver', 'kidney', 'spine',
                'rib cage', 'skull', 'cerebral cortex', 'ventricle',
                'pancreas', 'spleen', 'intestine', 'bladder', 'chest'
            ],
            'location_terms': [
                'left', 'right', 'anterior', 'posterior', 'superior',
                'inferior', 'lateral', 'medial', 'central', 'upper', 'lower',
                'bilateral', 'unilateral'
            ],
            'finding_terms': [
                'normal', 'abnormal', 'mass', 'lesion', 'fracture',
                'pneumonia', 'effusion', 'enlargement', 'tumor',
                'inflammation', 'infection', 'opacity', 'nodule'
            ],
            'measurement_terms': [
                'mm', 'cm', 'size', 'diameter', 'volume', 'density',
                'measurement', 'dimension'
            ],
            'comparison_terms': [
                'increased', 'decreased', 'unchanged', 'stable',
                'improved', 'worsened', 'compared to'
            ]
        }

        print("All models loaded successfully!")

    def evaluate_medical_quality(self, caption):
        """quality"""
        try:
            doc = self.nlp(caption)


            term_scores = {
                'modality': sum(term.lower() in caption.lower() for term in self.medical_patterns['modality_terms']),
                'anatomy': sum(term.lower() in caption.lower() for term in self.medical_patterns['anatomy_terms']),
                'location': sum(term.lower() in caption.lower() for term in self.medical_patterns['location_terms']),
                'finding': sum(term.lower() in caption.lower() for term in self.medical_patterns['finding_terms']),
                'measurement': sum(term.lower() in caption.lower() for term in self.medical_patterns['measurement_terms']),
                'comparison': sum(term.lower() in caption.lower() for term in self.medical_patterns['comparison_terms'])
            }


            medical_entities = []
            for ent in doc.ents:
                if ent._.kb_ents:
                    medical_entities.append(ent.text)


            word_count = len(caption.split())
            scores = {
                'term_usage': min(sum(term_scores.values()) / max(word_count, 1), 1.0),
                'entity_density': len(medical_entities) / max(word_count, 1),
                'term_diversity': len(set(medical_entities)) / max(len(medical_entities), 1)
            }


            medical_quality_score = (
                0.4 * scores['term_usage'] +
                0.3 * scores['entity_density'] +
                0.3 * scores['term_diversity']
            )

            return medical_quality_score, scores

        except Exception as e:
            print(f"Error in evaluate_medical_quality: {e}")
            return 0.0, {'term_usage': 0, 'entity_density': 0, 'term_diversity': 0}

    def evaluate_clinical_accuracy(self, caption):
        """acc"""
        try:
            # dignostic
            has_finding = any(term.lower() in caption.lower() for term in self.medical_patterns['finding_terms'])
            # anatomy
            has_location = any(term.lower() in caption.lower() for term in self.medical_patterns['location_terms'])
            has_anatomy = any(term.lower() in caption.lower() for term in self.medical_patterns['anatomy_terms'])

            has_measurements = any(term.lower() in caption.lower() for term in self.medical_patterns['measurement_terms'])

            has_comparison = any(term.lower() in caption.lower() for term in self.medical_patterns['comparison_terms'])

            scores = {
                'findings': float(has_finding),
                'location': float(has_location and has_anatomy),
                'measurement': float(has_measurements),
                'comparison': float(has_comparison)
            }

            clinical_accuracy_score = (
                0.375 * scores['findings'] +
                0.25 * scores['location'] +
                0.25 * scores['measurement'] +
                0.125 * scores['comparison']
            )

            return clinical_accuracy_score, scores

        except Exception as e:
            print(f"Error in evaluate_clinical_accuracy: {e}")
            return 0.0, {'findings': 0, 'location': 0, 'measurement': 0, 'comparison': 0}

    def evaluate_report_structure(self, caption):
        """structure"""
        try:

            has_modality = any(term.lower() in caption.lower() for term in self.medical_patterns['modality_terms'])
            has_anatomy = any(term.lower() in caption.lower() for term in self.medical_patterns['anatomy_terms'])
            basic_structure_score = float(has_modality and has_anatomy)


            sentences = [s.strip() for s in caption.split('.') if s.strip()]
            has_multiple_sentences = len(sentences) >= 2
            completeness_score = float(has_multiple_sentences)


            has_conclusion = bool(re.search(r'(suggest|indicate|represent|conclude|impression)', caption.lower()))
            logical_flow_score = float(has_conclusion)

            structure_score = (
                1/3 * basic_structure_score +
                1/3 * completeness_score +
                1/3 * logical_flow_score
            )

            scores = {
                'basic_structure': basic_structure_score,
                'completeness': completeness_score,
                'logical_flow': logical_flow_score
            }

            return structure_score, scores

        except Exception as e:
            print(f"Error in evaluate_report_structure: {e}")
            return 0.0, {'basic_structure': 0, 'completeness': 0, 'logical_flow': 0}

    def compute_similarities(self, image, caption, question):
        """similarity"""
        try:
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            caption_tokens = self.tokenizer(caption).to(self.device)
            question_tokens = self.tokenizer(question).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                caption_features = self.model.encode_text(caption_tokens)
                question_features = self.model.encode_text(question_tokens)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                caption_features /= caption_features.norm(dim=-1, keepdim=True)
                question_features /= question_features.norm(dim=-1, keepdim=True)

                image_text_score = torch.cosine_similarity(image_features, caption_features).item()
                question_text_score = torch.cosine_similarity(question_features, caption_features).item()

            return image_text_score, question_text_score

        except Exception as e:
            print(f"Error in compute_similarities: {e}")
            return 0.0, 0.0


    def compute_final_score(self, image_text_score, question_text_score, medical_quality_score,
                          clinical_accuracy_score, structure_score):
        """
new weight
        """
        try:

            relevance_score = (
                0.25 * image_text_score +
                0.25 * question_text_score
            )

            )
            quality_score = (
                (0.5/3) * medical_quality_score +
                (0.5/3) * clinical_accuracy_score +
                (0.5/3) * structure_score
            )


            final_score = relevance_score + quality_score


            if image_text_score < 0.2:
                final_score *= 0.5
            if question_text_score < 0.2:
                final_score *= 0.5

            return final_score, {
                'relevance_score': relevance_score,
                'quality_score': quality_score
            }

        except Exception as e:
            print(f"Error in compute_final_score: {e}")
            return 0.0, {'relevance_score': 0, 'quality_score': 0}


    def evaluate_caption(self, image, caption, question):
        """combine"""
        try:

            medical_quality_score, medical_quality_details = self.evaluate_medical_quality(caption)


            clinical_accuracy_score, clinical_accuracy_details = self.evaluate_clinical_accuracy(caption)


            structure_score, structure_details = self.evaluate_report_structure(caption)


            image_text_score, question_text_score = self.compute_similarities(image, caption, question)


            final_score, score_details = self.compute_final_score(
                image_text_score,
                question_text_score,
                medical_quality_score,
                clinical_accuracy_score,
                structure_score
            )


            return {
                'final_score': final_score,
                'medical_quality': medical_quality_score,
                'clinical_accuracy': clinical_accuracy_score,
                'structure': structure_score,
                'image_similarity': image_text_score,
                'question_similarity': question_text_score,
                'caption': caption,
                'question': question,
                'detailed_scores': {
                    'medical_quality': medical_quality_details,
                    'clinical_accuracy': clinical_accuracy_details,
                    'structure': structure_details,
                    'score_components': score_details
                }
            }

        except Exception as e:
            print(f"Error in evaluate_caption: {e}")
            return None

def display_case(case_data, img_path=None, case_type="", evaluator=None):
    """show cases"""
    try:
        print(f"\n{'='*20} {case_type} Case {'='*20}")


        print(f"Index/QID: {case_data['index']}")
        print(f"Image Name: {case_data['img_name']}")
        print(f"Modality: {case_data.get('modality', 'Unknown')}")
        print(f"Final Score: {case_data['final_score']:.4f}")


        print(f"\nQuestion: {case_data['question']}")
        if 'original_answer' in case_data:
            print(f"Original Answer: {case_data['original_answer']}")
        print(f"\nGenerated Caption: {case_data['caption']}")


        print("\n----- Component Scores -----")
        print(f"Medical Quality: {case_data['medical_quality']:.4f}")
        print(f"Clinical Accuracy: {case_data['clinical_accuracy']:.4f}")
        print(f"Structure: {case_data['structure']:.4f}")
        print(f"Image Similarity: {case_data['image_similarity']:.4f}")
        print(f"Question Similarity: {case_data['question_similarity']:.4f}")


        if 'detailed_scores' in case_data:
            print("\n----- Detailed Analysis -----")


            if 'medical_quality' in case_data['detailed_scores']:
                med_quality = case_data['detailed_scores']['medical_quality']
                print("\nMedical Quality Analysis:")
                print(f"  Term Usage: {med_quality.get('term_usage', 0):.4f}")
                print(f"  Entity Density: {med_quality.get('entity_density', 0):.4f}")
                print(f"  Term Diversity: {med_quality.get('term_diversity', 0):.4f}")


            if 'clinical_accuracy' in case_data['detailed_scores']:
                clin_acc = case_data['detailed_scores']['clinical_accuracy']
                print("\nClinical Accuracy Analysis:")
                print(f"  Findings: {clin_acc.get('findings', 0):.4f}")
                print(f"  Location: {clin_acc.get('location', 0):.4f}")
                print(f"  Measurement: {clin_acc.get('measurement', 0):.4f}")
                print(f"  Comparison: {clin_acc.get('comparison', 0):.4f}")


            if 'structure' in case_data['detailed_scores']:
                structure = case_data['detailed_scores']['structure']
                print("\nReport Structure Analysis:")
                print(f"  Basic Structure: {structure.get('basic_structure', 0):.4f}")
                print(f"  Completeness: {structure.get('completeness', 0):.4f}")
                print(f"  Logical Flow: {structure.get('logical_flow', 0):.4f}")


            if 'score_components' in case_data['detailed_scores']:
                components = case_data['detailed_scores']['score_components']
                print("\nScore Components:")
                print(f"  Relevance Score: {components.get('relevance_score', 0):.4f}")
                print(f"  Quality Score: {components.get('quality_score', 0):.4f}")


        if evaluator is not None:
            print("\n----- Keyword Analysis -----")
            caption_lower = case_data['caption'].lower()


            print("\nDetected Medical Terms:")
            for category, terms in [
                ("Modality Terms", evaluator.medical_patterns['modality_terms']),
                ("Anatomy Terms", evaluator.medical_patterns['anatomy_terms']),
                ("Location Terms", evaluator.medical_patterns['location_terms']),
                ("Finding Terms", evaluator.medical_patterns['finding_terms']),
                ("Measurement Terms", evaluator.medical_patterns['measurement_terms']),
                ("Comparison Terms", evaluator.medical_patterns['comparison_terms'])
            ]:
                found_terms = [term for term in terms if term.lower() in caption_lower]
                if found_terms:
                    print(f"  {category}: {', '.join(found_terms)}")


            question_keywords = [word.lower() for word in case_data['question'].split() if len(word) > 3]
            matching_keywords = [word for word in question_keywords if word in caption_lower]
            if matching_keywords:
                print("\nQuestion-Related Keywords Found:")
                print(f"  {', '.join(matching_keywords)}")


        if img_path and os.path.exists(img_path):
            print("\nDisplaying Image:")
            print(f"Image Path: {img_path}")
            image = Image.open(img_path).convert('RGB')
            display(image)
        else:
            print("\nImage not available or path not found.")

    except Exception as e:
        print(f"Error displaying case: {e}")
        print(traceback.format_exc())

def display_average_scores(df_results):
    """show case"""
    print("\n=== Average Scores ===")
    metrics = ['final_score', 'medical_quality', 'clinical_accuracy',
              'structure', 'image_similarity', 'question_similarity']

    for metric in metrics:
        mean_score = df_results[metric].mean()
        std_score = df_results[metric].std()
        print(f"{metric}:")
        print(f"  Mean: {mean_score:.4f}")
        print(f"  Std: {std_score:.4f}")

def evaluate_blip3_enhanced(caption_data, slake_base_dir, evaluator, model_name="BLIP3+Prompt"):
    """
show cases
    """
    try:
        print(f"\nEvaluating {model_name} results...")
        results = []

        for item in tqdm(caption_data, desc=f"Processing {model_name}"):
            try:

                qid = item.get('qid', 0)
                img_name = item.get('img_name', '')
                question = item.get('question', '')
                generated_caption = item.get('generated_caption', '')
                original_answer = item.get('original_answer', '')

                if not generated_caption:
                    print(f"Warning: Missing caption for qid {qid}")
                    continue


                img_path = os.path.join(slake_base_dir, 'imgs', img_name)


                if not os.path.exists(img_path):
                    print(f"Warning: Image not found at {img_path}")

                    alternative_paths = [
                        os.path.join(slake_base_dir, 'img', img_name),
                        os.path.join(slake_base_dir, 'images', img_name)
                    ]
                    for alt_path in alternative_paths:
                        if os.path.exists(alt_path):
                            img_path = alt_path
                            print(f"Found image at alternative path: {img_path}")
                            break
                    else:
                        print(f"Error: Could not find image for {img_name}")
                        continue


                image = Image.open(img_path).convert('RGB')


                result = evaluator.evaluate_caption(
                    image=image,
                    caption=generated_caption,
                    question=question
                )

                if result:

                    result['index'] = qid  # qid as index
                    result['img_name'] = img_name
                    result['img_path'] = img_path
                    result['modality'] = item.get('true_modality', '')
                    result['original_answer'] = original_answer
                    results.append(result)

            except Exception as e:
                print(f"Error processing item {qid}: {e}")
                print(traceback.format_exc())
                continue

        # DataFrame
        if results:
            df_results = pd.DataFrame([
                {
                    'index': r['index'],
                    'img_name': r['img_name'],
                    'img_path': r['img_path'],  # img_path
                    'caption': r['caption'],
                    'question': r['question'],
                    'modality': r['modality'],
                    'original_answer': r.get('original_answer', ''),
                    'final_score': r['final_score'],
                    'medical_quality': r['medical_quality'],
                    'clinical_accuracy': r['clinical_accuracy'],
                    'structure': r['structure'],
                    'image_similarity': r['image_similarity'],
                    'question_similarity': r['question_similarity'],
                    'detailed_scores': r['detailed_scores']
                }
                for r in results
            ])


            display_average_scores(df_results)


            top_n = 18
            top_indices = df_results['final_score'].nlargest(top_n).index
            print(f"\nDisplaying top {top_n} highest scoring cases:")

            for i, idx in enumerate(top_indices):
                case = df_results.loc[idx].to_dict()
                case_rank = i + 1
                display_case(case, case['img_path'], f"Rank {case_rank}: {model_name} Score {case['final_score']:.4f}", evaluator)


            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = '/content/drive/MyDrive/output'
            os.makedirs(output_dir, exist_ok=True)


            save_df = df_results.drop(columns=['detailed_scores'])
            output_path = os.path.join(output_dir, f'blip3_enhanced_evaluation_{timestamp}.csv')
            save_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")


            stats_path = os.path.join(output_dir, f'blip3_enhanced_evaluation_stats_{timestamp}.json')
            stats = {
                'average_scores': {
                    metric: {
                        'mean': float(df_results[metric].mean()),
                        'std': float(df_results[metric].std()),
                        'min': float(df_results[metric].min()),
                        'max': float(df_results[metric].max())
                    }
                    for metric in ['final_score', 'medical_quality', 'clinical_accuracy',
                                 'structure', 'image_similarity', 'question_similarity']
                },
                'sample_count': len(df_results),
                'top_case_indices': [int(idx) for idx in top_indices[:5]]
            }

            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)
            print(f"Statistics saved to: {stats_path}")

            return df_results

        else:
            print(f"No valid results for {model_name}")
            return None

    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        print(traceback.format_exc())
        return None

def evaluate_blip3_basic(caption_data, slake_base_dir, evaluator, model_name="BLIP3"):
    """
blip3
    """
    try:
        print(f"\nEvaluating {model_name} results...")
        results = []

        for item in tqdm(caption_data, desc=f"Processing {model_name}"):
            try:

                qid = item.get('qid', 0)
                img_name = item.get('img_name', '')
                question = item.get('question', '')
                caption_text = item.get('caption', '')
                original_answer = item.get('answer', '')  # answer

                if not caption_text:
                    print(f"Warning: Missing caption for qid {qid}")
                    continue


                img_path = os.path.join(slake_base_dir, 'imgs', img_name)


                if not os.path.exists(img_path):
                    print(f"Warning: Image not found at {img_path}")

                    alternative_paths = [
                        os.path.join(slake_base_dir, 'img', img_name),
                        os.path.join(slake_base_dir, 'images', img_name)
                    ]
                    for alt_path in alternative_paths:
                        if os.path.exists(alt_path):
                            img_path = alt_path
                            print(f"Found image at alternative path: {img_path}")
                            break
                    else:
                        print(f"Error: Could not find image for {img_name}")
                        continue


                image = Image.open(img_path).convert('RGB')


                result = evaluator.evaluate_caption(
                    image=image,
                    caption=caption_text,
                    question=question
                )

                if result:

                    result['index'] = qid  # qid as index
                    result['img_name'] = img_name
                    result['img_path'] = img_path
                    result['modality'] = item.get('modality', '')
                    result['original_answer'] = original_answer
                    results.append(result)

            except Exception as e:
                print(f"Error processing item {qid}: {e}")
                print(traceback.format_exc())
                continue

        # DataFrame
        if results:
            df_results = pd.DataFrame([
                {
                    'index': r['index'],
                    'img_name': r['img_name'],
                    'img_path': r['img_path'],
                    'caption': r['caption'],
                    'question': r['question'],
                    'modality': r['modality'],
                    'original_answer': r.get('original_answer', ''),
                    'final_score': r['final_score'],
                    'medical_quality': r['medical_quality'],
                    'clinical_accuracy': r['clinical_accuracy'],
                    'structure': r['structure'],
                    'image_similarity': r['image_similarity'],
                    'question_similarity': r['question_similarity'],
                    'detailed_scores': r['detailed_scores']
                }
                for r in results
            ])


            display_average_scores(df_results)


            top_n = 18
            top_indices = df_results['final_score'].nlargest(top_n).index
            print(f"\nDisplaying top {top_n} highest scoring cases:")

            for i, idx in enumerate(top_indices):
                case = df_results.loc[idx].to_dict()
                case_rank = i + 1
                display_case(case, case['img_path'], f"Rank {case_rank}: {model_name} Score {case['final_score']:.4f}", evaluator)


            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = '/content/drive/MyDrive/output'
            os.makedirs(output_dir, exist_ok=True)


            save_df = df_results.drop(columns=['detailed_scores'])
            output_path = os.path.join(output_dir, f'blip3_basic_evaluation_{timestamp}.csv')
            save_df.to_csv(output_path, index=False)
            print(f"\nResults saved to: {output_path}")


            stats_path = os.path.join(output_dir, f'blip3_basic_evaluation_stats_{timestamp}.json')
            stats = {
                'average_scores': {
                    metric: {
                        'mean': float(df_results[metric].mean()),
                        'std': float(df_results[metric].std()),
                        'min': float(df_results[metric].min()),
                        'max': float(df_results[metric].max())
                    }
                    for metric in ['final_score', 'medical_quality', 'clinical_accuracy',
                                 'structure', 'image_similarity', 'question_similarity']
                },
                'sample_count': len(df_results),
                'top_case_indices': [int(idx) for idx in top_indices[:5]]
            }

            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=4)
            print(f"Statistics saved to: {stats_path}")

            return df_results

        else:
            print(f"No valid results for {model_name}")
            return None

    except Exception as e:
        print(f"Error evaluating {model_name}: {e}")
        print(traceback.format_exc())
        return None
def main_blip3_evaluation():
    try:
        
        slake_base_dir = '/content/drive/MyDrive/slakedataset/Slake1.0'
        enhanced_path = '/content/drive/MyDrive/output/enhanced_captions_wholeslake.json'
        basic_path = '/content/drive/MyDrive/output/test_captions_slake_blip3_full_20250224_110939.json'

        
        evaluator = MedicalImageCaptionEvaluator()

        # BLIP3+Prompt
        print("\nProcessing BLIP3+Prompt results...")
        try:
            with open(enhanced_path, 'r') as f:
                enhanced_data = json.load(f)

            if 'results' in enhanced_data:

                enhanced_results = enhanced_data['results']
            else:

                enhanced_results = enhanced_data

            print(f"Loaded {len(enhanced_results)} BLIP3+Prompt captions")
            enhanced_df = evaluate_blip3_enhanced(enhanced_results, slake_base_dir, evaluator, "BLIP3+Prompt")
        except Exception as e:
            print(f"Error processing BLIP3+Prompt results: {e}")
            print(traceback.format_exc())

        # BLIP3
        print("\nProcessing BLIP3 results...")
        try:
            with open(basic_path, 'r') as f:
                basic_data = json.load(f)

            if 'results' in basic_data:

                basic_results = basic_data['results']
            else:

                basic_results = basic_data

            print(f"Loaded {len(basic_results)} BLIP3 captions")
            basic_df = evaluate_blip3_basic(basic_results, slake_base_dir, evaluator, "BLIP3")
        except Exception as e:
            print(f"Error processing BLIP3 results: {e}")
            print(traceback.format_exc())

        print("\nEvaluation complete!")

    except Exception as e:
        print(f"Error in main_blip3_evaluation: {e}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main_blip3_evaluation()



