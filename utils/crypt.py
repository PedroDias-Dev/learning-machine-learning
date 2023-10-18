def encrypt(text, key):
    encrypted_text = ''
    for char in text:
        if char.isalpha():
            char_offset = ord('A') if char.isupper() else ord('a')
            shifted_char = chr(char_offset + (ord(char) - char_offset + key) % 26)
            encrypted_text += shifted_char
        else:
            encrypted_text += char
    return encrypted_text

def decrypt(encrypted_text, key):
    return encrypt(encrypted_text, -key)
