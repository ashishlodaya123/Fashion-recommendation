# Occasion and Color Aware Personalized Outfit Recommendation System

This project introduces an innovative fashion recommendation system designed to provide personalized outfit suggestions. By incorporating occasion awareness, user-defined color preferences, and outfit compatibility validation, this system enhances user satisfaction through cutting-edge machine learning and computer vision techniques.

## Features

- **Personalized Recommendations**: Users receive outfit suggestions tailored to specific occasions and preferred colors.
- **Advanced Color Matching**: Utilizes K-means clustering for dominant color extraction and Euclidean distance for precise color matching.
- **Outfit Compatibility**: Employs conditional Generative Adversarial Networks (cGANs) to ensure visual and aesthetic harmony across ensembles.
- **Interactive Interface**: Natural Language Processing (NLP) allows users to input preferences intuitively, e.g., "Recommend a blue casual outfit."

## System Architecture

The system consists of several modular components:

1. **User Interaction Layer**: Captures user inputs via natural language.
2. **Data Retrieval Layer**: Fetches and aligns clothing item data and images.
3. **Preprocessing Layer**: Filters data based on user-specified criteria (occasion, color).
4. **Recommendation Engine**: Assembles compatible outfits using scoring algorithms.
5. **Compatibility Validation**: Refines recommendations using cGANs.
6. **Output and Visualization Layer**: Delivers detailed outfit suggestions with high-resolution visuals.

## Methodology

### Dataset
The system is built on a curated fashion dataset with metadata including dominant colors, categories, and occasions.

### Key Techniques
- **Dominant Color Extraction**: K-means clustering identifies primary colors in images.
- **Color Matching**: Euclidean distance ensures precise alignment with user preferences.
- **Outfit Validation**: cGANs generate and verify aesthetically coherent combinations.

## Performance Metrics

- **Accuracy**: 94.61% compatibility accuracy.
- **Precision**: 92.82%.
- **Recall**: 100%.
- **F1-Score**: 96.27%.

## Experimental Results

The system was tested on diverse user inputs, including specific occasions and colors, achieving superior performance compared to existing models in the domain.

## Future Directions

- Expand the dataset for broader inclusivity and cultural representation.
- Incorporate real-time user feedback for dynamic and adaptive recommendations.
- Explore hybrid models combining collaborative filtering with generative techniques.

## Authors

- Narasimha Shastry B K, Aditi Ponkshe, Ashish Lodaya,Anjana Bharamnaikar 
- **Affiliation**: KLE Technological University, Hubballi, India


## License

This project is open-source and available under the [MIT License](LICENSE).
