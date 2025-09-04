"""
Demo script to show how to run the Forest Cover Type Classification project
"""

import subprocess
import sys
import os

def run_streamlit_app():
    """Run the Streamlit application."""
    print("🌲 Starting Forest Cover Type Classification - Streamlit App")
    print("=" * 60)
    print("The app will open in your default web browser.")
    print("If it doesn't open automatically, go to: http://localhost:8501")
    print("=" * 60)
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running Streamlit app: {e}")
    except KeyboardInterrupt:
        print("\nStreamlit app stopped by user.")

def show_instructions():
    """Show instructions for running the project."""
    print("🌲 Forest Cover Type Classification Project")
    print("=" * 50)
    print()
    print("This project includes two main components:")
    print()
    print("1. 📓 JUPYTER NOTEBOOK (forest_cover_classification.ipynb)")
    print("   - Detailed machine learning analysis")
    print("   - Data exploration and visualization")
    print("   - Model training and evaluation")
    print("   - Hyperparameter tuning")
    print("   - Feature importance analysis")
    print()
    print("   To run: jupyter notebook forest_cover_classification.ipynb")
    print()
    print("2. 🌐 STREAMLIT WEB APP (app.py)")
    print("   - Interactive web interface")
    print("   - Data exploration with visualizations")
    print("   - Model training with one click")
    print("   - Single and batch predictions")
    print("   - CSV file upload and download")
    print("   - Results analysis and visualization")
    print()
    print("   To run: streamlit run app.py")
    print()
    print("=" * 50)
    print("Choose an option:")
    print("1. Run Streamlit Web App")
    print("2. Show instructions only")
    print("3. Exit")
    print("=" * 50)

def main():
    """Main function."""
    show_instructions()
    
    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()
            
            if choice == "1":
                run_streamlit_app()
                break
            elif choice == "2":
                show_instructions()
            elif choice == "3":
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
