with open ("SPY ETF Stock Price History 93-25.csv", 'r') as file:
    lines = file.readlines()
    lines = [line.strip() for line in lines] 

    print(lines[0:10])