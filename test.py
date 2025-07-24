import pickle 
with open("shared_variable.pkl", "rb") as f:
    data = pickle.load(f)

# Access each variable
a = data['fdlty']
b = data['counta']
print(a)
print(b)
