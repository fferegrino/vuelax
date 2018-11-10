def create_trip(tokens, predicted_labels):
    destination_tokens = []
    origin_tokens = []
    price = -1
    for token, label in zip(tokens, predicted_labels):
        if label == 'o':
            origin_tokens.append(token)
        elif label == 'd':
            destination_tokens.append(token)
        elif label == 'p':
            price = float(token.replace(',', ''))
    return {
        'origin': ' '.join(origin_tokens),
        'destination': ' '.join(destination_tokens),
        'price': price
    }
