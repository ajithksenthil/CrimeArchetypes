def process_and_update_matrices(events, transition_matrix):
    time_series_matrix = []
    
    for i in range(len(events) - 1):
        prev_state = categorize_life_event(events[i]['event'])
        next_state = categorize_life_event(events[i + 1]['event'])
        
        prev_vector = cog_vectors[prev_state]
        next_vector = cog_vectors[next_state]
        
        transition_matrix[prev_vector][next_vector] += 1
        
        time_series_matrix.append(transition_matrix.copy())
    
    return time_series_matrix, transition_matrix

# Initialize transition matrix
transition_matrix = np.zeros((NUM_COG_VECTORS, NUM_COG_VECTORS))

# Process events and update matrices
events = read_csv_life_events('Type1_Allanson_Patricia.csv')
time_series_matrix, final_matrix = process_and_update_matrices(events, transition_matrix)

# Normalize the final matrix
row_sums = final_matrix.sum(axis=1)
normalized_matrix = final_matrix / row_sums[:, np.newaxis]

# Save individual killer's matrix
np.save(f'transition_matrix_Allanson_Patricia.npy', normalized_matrix)

# Add to the aggregate matrix (you'll need to initialize this earlier if processing multiple killers)
aggregate_matrix += normalized_matrix

# After processing all killers, normalize and save the aggregate matrix
aggregate_matrix /= len(killer_list)  # Assuming you have a list of all killers processed
np.save('aggregate_transition_matrix.npy', aggregate_matrix)
