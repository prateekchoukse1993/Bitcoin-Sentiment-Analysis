#Import the necessary methods from tweepy library

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

#Variables that contains the user credentials to access Twitter API 
access_token = "2895507294-nVZ9polODDRn5ATYyWFLYybLUtc0QtFUtDDxkXA"
access_token_secret = "OUSPqaLQt40lQ4J3EH4dn7aaRXewuo9T09e8vBVMntF41"
consumer_key = "aFzfJhYabqraaCxjANsT7ysTa"
consumer_secret = "p25ek776zJ14M2AjCS4nKzAw7mb9QAZwm5sgoGb0bKhgFvlFKD"


#This is a basic listener that just prints received tweets to stdout.

class StdOutListener(StreamListener):

    def on_data(self, data):
        print(data)
        return True

    def on_error(self, status):
        print(status)


if __name__ == '__main__':

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    stream.filter(languages=["en"], track=['bitcoin','bitcoins','btc','#btc','#bitcoin','#bitcointalk', '#bitcoins'])