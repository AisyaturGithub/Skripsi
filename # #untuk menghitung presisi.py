 #                         #untuk menghitung presisi
                # def calculate_precision(test_data, dictionary):
                #     true_positives = 0
                #     false_positives = 0

                #     for test_word, true_word in test_data:
                #         corrected_word = correct_spelling(test_word, dictionary)
                #         if corrected_word == true_word:
                #             true_positives += 1
                #         else:
                #             false_positives += 1

                #     if true_positives + false_positives == 0:
                #         return 0.0
                #     else:
                #         precision = true_positives / (true_positives + false_positives)
                #         return precision

                # # def calculate_precision_with_dictionary(input_sentence, dictionary):
                # #     input_words = input_sentence.split()
    
                # #     # Hitung jumlah kata yang ada dalam kamus
                # #     correct_predictions = sum(1 for word in input_words if word in dictionary)
                    
                # #     # Hitung jumlah total kata yang diperiksa
                # #     total_predictions = len(input_words)
                    
                # #     # Hitung nilai presisi
                # #     if total_predictions == 0:
                # #         return 0.0
                # #     else:
                # #         precision = correct_predictions / total_predictions
                    
                # #     return precision
                
                # # precision = calculate_precision_with_dictionary(input_sentence, dictionary)
                # # print(f"Nilai presisi: {precision:.2f}")
