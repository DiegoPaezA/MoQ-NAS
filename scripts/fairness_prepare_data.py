# moq-nas/scripts/fairness_prepare_data.py
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# We import the build functions from our fairness module
from core.fairness.build import (
    build_facet_csv, 
    build_face_binary_dataset, 
    build_person_binary_dataset
)

def main():
    """
    Main script to orchestrate the construction of all datasets
    required for the fairness analysis pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Prepares all datasets for fairness training and evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Arguments for the PERSON dataset (from COCO) ---
    parser.add_argument(
        '--build_person', action='store_true', 
        help="Enables the build process for the PERSON/NON_PERSON dataset."
    )
    parser.add_argument(
        '--coco_root', type=str, default='datasets/COCO_sub', 
        help="Path to the COCO root directory (containing train2017, val2017, annotations)."
    )
    parser.add_argument(
        '--person_out_dir', type=str, default='datasets/personbin_data',
        help="Output directory for the PERSON/NON_PERSON dataset."
    )

    # --- Arguments for the FACE dataset (from WIDER) ---
    parser.add_argument(
        '--build_face', action='store_true', 
        help="Enables the build process for the FACE/NON_FACE dataset."
    )
    parser.add_argument(
        '--wider_root', type=str, default='datasets/WIDER', 
        help="Path to the WIDER Face root directory."
    )
    parser.add_argument(
        '--places_root', type=str, default='datasets/PLACES365/val', 
        help="Path to a directory of negative images (e.g., Places365) for the face dataset."
    )
    parser.add_argument(
        '--face_out_dir', type=str, default='datasets/facebin_data', 
        help="Output directory for the FACE/NON_FACE dataset."
    )

    # --- Arguments for the FACET CSV ---
    parser.add_argument(
        '--build_facet', action='store_true', 
        help="Enables the build process for the FACET evaluation CSV."
    )
    parser.add_argument(
        '--facet_ann', type=str, default='datasets/facet_data/annotations/annotations.csv', 
        help="Path to the raw FACET annotations CSV file."
    )
    parser.add_argument(
        '--facet_img_dirs', nargs='+', default=['datasets/facet_data/imgs_1', 'datasets/facet_data/imgs_2'], 
        help="A list of directories containing the FACET images."
    )
    parser.add_argument(
        '--facet_out_csv', type=str, default='datasets/facet_data/facet_eval.csv', 
        help="Output path for the processed FACET CSV."
    )
    
    args = parser.parse_args()

    # --- Task Execution ---
    if not any([args.build_person, args.build_face, args.build_facet]):
        print("\nNo build action was specified. Use --build_person, --build_face, or --build_facet to begin.")
        print("Example: python scripts/fairness_prepare_data.py --build_person --build_face --build_facet")
        return

    if args.build_person:
        print("\n--- Starting PERSON/NON_PERSON dataset build ---")
        build_person_binary_dataset(
            coco_root=args.coco_root,
            out_dir=args.person_out_dir
            # You could pass more kwargs here if needed, e.g., seed=42
        )
        print("--- Finished: PERSON/NON_PERSON ---")

    if args.build_face:
        print("\n--- Starting FACE/NON_FACE dataset build ---")
        build_face_binary_dataset(
            wider_root=args.wider_root,
            neg_root=args.places_root,
            out_dir=args.face_out_dir
        )
        print("--- Finished: FACE/NON_FACE ---")
        
    if args.build_facet:
        print("\n--- Starting FACET CSV build ---")
        build_facet_csv(
            ann_csv=args.facet_ann,
            img_dirs=args.facet_img_dirs,
            out_csv=args.facet_out_csv
        )
        print("--- Finished: FACET CSV ---")

if __name__ == "__main__":
    main()