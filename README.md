GESTURE RECOGNITION

This project aim to recognize live some hand gesture using a trained Recurrent Neural Network (RNN) model.
It leverages hand landmark detection to extract meaningful features from video frames and classify gestures in a live webcam feed.

### Prerequisites
Before running the application, ensure you have the following installed:
* Python 3.x
* `pip` (Python package installer)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1.  **Ensure your webcam is connected and accessible.**
2.  **Run the real-time detection script:**
    ```bash
    python inference.py
    ```
    A window titled "Wave Detection" will open, displaying your webcam feed with real-time gesture predictions.
3.  **Perform gestures:**
    * Try performing some gestures (e.g., waving, holding a still hand, swiping).
    * The detected gesture and its confidence will be displayed on the screen.
4.  **Exit the application:**
    * Press the `q` key on your keyboard.
    * Close the display window using the 'X' button.
