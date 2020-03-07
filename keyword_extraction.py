from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener

access_key = ''
access_secret = ''
consumer_key = ''
consumer_secret = ''


class listener(StreamListener):

    def on_data(self, data):
        savefile=open('saras.txt','a')
        savefile.write(data)
        savefile.write('\n')
        savefile.close()
        return True
        print(data)
    def on_error(self, status):
        print status

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_key, access_secret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["#saraswati"])  # the hashtag you want 




		
