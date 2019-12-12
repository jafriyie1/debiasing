import praw 
import argparse 
import re
import pandas as pd
import os
from nltk.tokenize import word_tokenize

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--name', default='all', type=str)
    p = args.parse_args()



    reddit = praw.Reddit(client_id='ksThlLQkJh0Tfw', 
                            client_secret='35TwVGpsH_sToxPyax3SYJipMeE', 
                            user_agent='Get-Data')

    df_list = []
    hot_posts = reddit.subreddit(p.name).top(limit=100)
    for post in hot_posts:
        title = post.title
        post.comments.replace_more(limit=0)
        for s in post.comments:
            comment = s.body
            n_str = re.sub('(\.|\?|\!)', r'\1|', comment)
            sent = n_str.split("|")

            for n_str in sent:

                n_str = n_str.replace('\n', '').replace('\r', '').replace('\t', '')
                n_str = n_str.strip()
                

                if len(n_str) < 5 or 'http' \
                    in n_str or 'https' \
                    in n_str or '/com' in n_str \
                    in n_str or 'com/' in n_str\
                    in n_str or '/r' in n_str \
                    in n_str or '/r' in n_str \
                    in n_str or 'be/' in n_str \
                    in n_str or '=' in n_str:
                    continue 
                else:
                    print(n_str)
                    d = {'Sentence': n_str, 'Post': title, 'Subreddit': p.name}
                    df_list.append(d)

    df = pd.DataFrame(df_list)
    n_path =os.path.join(os.getcwd(), 'reddit_data', 'reddit_csv_from_'+p.name+'.csv')
    df.to_csv(n_path, index=False)
    print('Done')
    


