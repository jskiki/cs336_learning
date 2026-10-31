print(chr(0))
print("this is a test" + chr(0) + "string")

test_string = "hello!,你好！"
utf8_encoder = test_string.encode('utf-8')
print(utf8_encoder) 
#b'hello!,\xe4\xbd\xa0\xe5\xa5\xbd\xef\xbc\x81'
#hello也变成字节了，只是在打印时python会把可打印的 ASCII 字节直接显示为字符

list(utf8_encoder)
print(len(test_string))
print(len(utf8_encoder))