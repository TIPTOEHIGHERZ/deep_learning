import time
from tqdm import tqdm


pbar = tqdm(range(10))
for index, i in enumerate(pbar):
    time.sleep(0.1)
    if index == 9:
        pbar.set_postfix({'status': 1})

