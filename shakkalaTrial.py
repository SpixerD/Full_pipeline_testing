from shakkala import Shakkala

# Initialize the Shakkala diacritizer
sh = Shakkala()

# Input Arabic text without diacritics
input_text = "اللغة العربية جميلة"

# Prepare the input for the model
input_int = sh.prepare_input(input_text)

# Get the model. The 'graph' object might not be needed for prediction in TF2.
returned_from_get_model = sh.get_model()

# Determine if get_model() returns a tuple (model, graph) or just the model
if isinstance(returned_from_get_model, tuple) and len(returned_from_get_model) > 0:
    model = returned_from_get_model[0]
    # graph = returned_from_get_model[1] # graph might not be used
else:
    model = returned_from_get_model

# Predict diacritics (remove the `with graph.as_default():` block)
# Keras models in TF2 typically run .predict eagerly.
print("start load model") # This was in your log, adding it back for reference
logits = model.predict(input_int)[0]
print("end load model") # This was in your log

# Convert logits to diacritics
predicted_harakat = sh.logits_to_text(logits)

# Combine original text with predicted diacritics
final_output = sh.get_final_text(input_text, predicted_harakat)

print(final_output)