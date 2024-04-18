import argparse
from room_corner_model_analyzer import RoomCornerModelAnalyzer

def main():
    parser = argparse.ArgumentParser(description='Analyze room corner models from a JSON file.')
    parser.add_argument('url', type=str, help='URL to the JSON file containing the data.') 
    args = parser.parse_args()
    
    paths = RoomCornerModelAnalyzer.draw_plots(args.url)
    for path in paths: 
        print(path)

if __name__ == "__main__":
    main()
