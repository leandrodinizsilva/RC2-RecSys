import pandas as pd
import sys

Ratings = pd.read_json(sys.argv[1], lines=True)
Content = pd.read_json(sys.argv[2], lines=True)
Target = pd.read_csv(sys.argv[3])