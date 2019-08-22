import os
import base64

class Encrypt:
    def __init__(self):
        pass

    @staticmethod
    def verify_encryption_key():
        if not 'ENCRYPTION_KEY' in os.environ:
            raise Exception('Please set the encryption key as an environment variable (`ENCRYPTION_KEY`) first.')

    @property
    def key(self):
        key = os.environ.get('ENCRYPTION_KEY', None)
        if key is None:
            raise Exception('Please set the encryption key as an environment variable (`ENCRYPTION_KEY`) first.')
        return key

    def encode(self, clear):
        enc = []
        for i in range(len(clear)):
            key_c = self.key[i % len(self.key)]
            enc_c = chr((ord(clear[i]) + ord(key_c)) % 256)
            enc.append(enc_c)
        return base64.urlsafe_b64encode("".join(enc).encode()).decode()

    def decode(self, enc):
        dec = []
        enc = base64.urlsafe_b64decode(enc).decode()
        for i in range(len(enc)):
            key_c = self.key[i % len(self.key)]
            dec_c = chr((256 + ord(enc[i]) - ord(key_c)) % 256)
            dec.append(dec_c)
        return "".join(dec)

    def encode_tweet(self, tweet, encrypt_fields):
        for k, dtype in encrypt_fields.items():
            if k in tweet and tweet[k] is not None:
                if dtype == str:
                    tweet[k] = self.encode(tweet[k])
                elif dtype == int:
                    tweet[k] = self.encode(str(tweet[k]))
                elif dtype == list:
                    encrypted_items = []
                    for item in tweet[k]:
                        if item is None:
                            continue
                        encrypted_items.append(self.encode(item))
                    if len(encrypted_items) > 0:
                        tweet[k] = encrypted_items
                    else:
                        tweet[k] = None
        return tweet

    def decode_tweet(self, tweet, encrypt_fields):
        for k, dtype in encrypt_fields.items():
            if k in tweet and tweet[k] is not None:
                if dtype == str:
                    tweet[k] = self.decode(tweet[k])
                elif dtype == int:
                    tweet[k] = int(self.decode(tweet[k]))
                elif dtype == list:
                    decrypted_items = []
                    for item in tweet[k]:
                        decrypted_items.append(self.decode(item))
                    if len(decrypted_items) > 0:
                        tweet[k] = decrypted_items
                    else:
                        tweet[k] = None
        return tweet
