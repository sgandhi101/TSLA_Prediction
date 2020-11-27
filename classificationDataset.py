import pandas as pd

data = pd.read_csv('aggregate.csv')
nas = pd.read_csv('nasdaq.csv')

for i in range(1, len(data)):
    def determine():
        if data.loc[i, 'close'] - data.loc[i - 1, 'close'] > 0:
            return True
        else:
            return False
    data.loc[i, 'positiveChange'] = determine()

data = data.drop(0)
print(data)

for i in range(1, len(nas)):
    def determine():
        if nas.loc[i, 'nasClose'] - nas.loc[i - 1, 'nasClose'] > 0:
            return True
        else:
            return False
    nas.loc[i, 'nasChange'] = determine()
print(nas)
del nas['nasClose']
data = pd.merge(data, nas, on='date')

data.to_csv(r'classificationAggregate.csv', index=False, header=True)
