#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

constexpr int MASK_COUNT = 7;
constexpr int KEY_COUNT_PER_STATE = 3;
constexpr int GRADE_MASK_COUNT = 8;
constexpr int MAX_GRADE_TRANSITIONS_PER_MASK = 8;
constexpr int SINGLE_SLOT_KEY_SIZE = 20;
constexpr int TWO_SLOT_KEY_SIZE = SINGLE_SLOT_KEY_SIZE * SINGLE_SLOT_KEY_SIZE;
constexpr double COMPARISON_EPSILON = 1e-12;

constexpr std::array<int, MASK_COUNT> SINGLE_MASK_INDEX = {-1, 0, 1, -1, 2, -1, -1};
constexpr std::array<int, MASK_COUNT> PAIR_MASK_INDEX = {-1, -1, -1, 0, -1, 1, 2};
constexpr std::array<int, 3> SINGLE_WEIGHT_INDEX_BY_SLOT = {1, 2, 4};
constexpr std::array<int, 3> PAIR_WEIGHT_INDEX_BY_SLOT = {3, 5, 6};

struct ValueAggregates {
  std::array<std::vector<double>, 3> single_sums;
  std::array<std::vector<double>, 3> pair_sums;
  double all_sum = 0.0;

  ValueAggregates() {
    for (int slot = 0; slot < 3; ++slot) {
      single_sums[slot].assign(SINGLE_SLOT_KEY_SIZE, 0.0);
      pair_sums[slot].assign(TWO_SLOT_KEY_SIZE, 0.0);
    }
  }

  void reset() {
    all_sum = 0.0;
    for (int slot = 0; slot < 3; ++slot) {
      std::fill(single_sums[slot].begin(), single_sums[slot].end(), 0.0);
      std::fill(pair_sums[slot].begin(), pair_sums[slot].end(), 0.0);
    }
  }
};

double get_aggregated_value_sum(
  int protected_mask,
  int state_index,
  const int32_t* single_compatibility_keys,
  const int32_t* pair_compatibility_keys,
  const ValueAggregates& aggregates
) {
  const int key_offset = state_index * KEY_COUNT_PER_STATE;
  if (protected_mask == 0) {
    return aggregates.all_sum;
  }

  const int single_slot = SINGLE_MASK_INDEX[protected_mask];
  if (single_slot != -1) {
    return aggregates.single_sums[single_slot][single_compatibility_keys[key_offset + single_slot]];
  }

  const int pair_slot = PAIR_MASK_INDEX[protected_mask];
  if (pair_slot == -1) {
    return 0.0;
  }

  return aggregates.pair_sums[pair_slot][pair_compatibility_keys[key_offset + pair_slot]];
}

bool is_better_candidate(
  double probability,
  double expected_lock_key_cost,
  double best_probability,
  double best_expected_lock_key_cost
) {
  if (probability > best_probability + COMPARISON_EPSILON) {
    return true;
  }

  if (std::abs(probability - best_probability) <= COMPARISON_EPSILON) {
    return expected_lock_key_cost + COMPARISON_EPSILON < best_expected_lock_key_cost;
  }

  return false;
}

}  // namespace

extern "C" {

void run_budget_optimizer_core(
  int32_t state_count,
  int32_t module_budget,
  const int8_t* state_module_masks,
  const int32_t* single_compatibility_keys,
  const int32_t* pair_compatibility_keys,
  const int32_t* next_state_index_by_module_mask_and_grade_mask,
  const uint8_t* target_state_by_index,
  const double* option_transition_weights_by_state,
  const uint8_t* grade_transition_counts_by_state_and_mask,
  const uint8_t* grade_transition_grade_masks_by_state_and_mask,
  const double* grade_transition_probabilities_by_state_and_mask,
  const int8_t* action_module_cost_by_mask_triplet,
  const int8_t* action_lock_key_cost_by_mask_triplet,
  const uint8_t* option_candidate_counts_by_state,
  const int8_t* option_candidate_next_module_masks_by_state,
  const int8_t* option_candidate_protected_masks_by_state,
  const int8_t* option_candidate_key_masks_by_state,
  const double* option_candidate_probability_masses_by_state,
  const uint8_t* grade_candidate_counts_by_state,
  const int8_t* grade_candidate_next_module_masks_by_state,
  const int8_t* grade_candidate_protected_masks_by_state,
  const int8_t* grade_candidate_key_masks_by_state,
  double* probability_table,
  double* expected_lock_key_table,
  int8_t* action_type_table,
  int8_t* action_module_mask_table,
  int8_t* action_key_mask_table
) {
  auto build_action_cost_index = [](int current_module_mask, int next_module_mask, int key_mask) {
    return (current_module_mask * MASK_COUNT + next_module_mask) * MASK_COUNT + key_mask;
  };

  std::vector<std::array<ValueAggregates, MASK_COUNT>> probability_aggregates_cache(module_budget + 1);
  std::vector<std::array<ValueAggregates, MASK_COUNT>> lock_key_aggregates_cache(module_budget + 1);
  std::vector<uint8_t> probability_aggregates_built(module_budget + 1);
  std::vector<uint8_t> lock_key_aggregates_built(module_budget + 1);

  auto build_aggregates_for_budget = [&](int remaining_budget, const double* table, std::array<ValueAggregates, MASK_COUNT>& by_module_mask) {
    for (int module_mask = 0; module_mask < MASK_COUNT; ++module_mask) {
      by_module_mask[module_mask].reset();
    }

    const int base_offset = remaining_budget * state_count;
    for (int state_index = 0; state_index < state_count; ++state_index) {
      const double value = table[base_offset + state_index];
      if (value == 0.0) {
        continue;
      }

      auto& aggregates = by_module_mask[state_module_masks[state_index]];
      const int weight_offset = state_index * MASK_COUNT;
      const int key_offset = state_index * KEY_COUNT_PER_STATE;
      aggregates.all_sum += option_transition_weights_by_state[weight_offset] * value;

      for (int slot = 0; slot < 3; ++slot) {
        const int single_key = single_compatibility_keys[key_offset + slot];
        const double weight = option_transition_weights_by_state[weight_offset + SINGLE_WEIGHT_INDEX_BY_SLOT[slot]];
        aggregates.single_sums[slot][single_key] += weight * value;
      }

      for (int pair_index = 0; pair_index < 3; ++pair_index) {
        const int pair_key = pair_compatibility_keys[key_offset + pair_index];
        const double weight = option_transition_weights_by_state[weight_offset + PAIR_WEIGHT_INDEX_BY_SLOT[pair_index]];
        aggregates.pair_sums[pair_index][pair_key] += weight * value;
      }
    }
  };

  for (int budget = 0; budget <= module_budget; ++budget) {
    std::fill(probability_aggregates_built.begin(), probability_aggregates_built.begin() + budget + 1, 0);
    std::fill(lock_key_aggregates_built.begin(), lock_key_aggregates_built.begin() + budget + 1, 0);

    for (int state_index = 0; state_index < state_count; ++state_index) {
      const int table_index = budget * state_count + state_index;

      if (target_state_by_index[state_index] != 0) {
        probability_table[table_index] = 1.0;
        expected_lock_key_table[table_index] = 0.0;
        action_type_table[table_index] = 0;
        action_module_mask_table[table_index] = 0;
        action_key_mask_table[table_index] = 0;
        continue;
      }

      const int current_module_mask = state_module_masks[state_index];
      double best_probability = 0.0;
      double best_expected_lock_key_cost = std::numeric_limits<double>::infinity();
      int best_action_type = -1;
      int best_module_mask = -1;
      int best_key_mask = -1;
      const int candidate_offset = state_index * 19;

      for (int candidate_index = 0; candidate_index < option_candidate_counts_by_state[state_index]; ++candidate_index) {
        const int action_index = candidate_offset + candidate_index;
        const int next_module_mask = option_candidate_next_module_masks_by_state[action_index];
        const int protected_mask = option_candidate_protected_masks_by_state[action_index];
        const int key_mask = option_candidate_key_masks_by_state[action_index];
        const int action_cost_index = build_action_cost_index(current_module_mask, next_module_mask, key_mask);
        const int module_cost = action_module_cost_by_mask_triplet[action_cost_index];
        if (module_cost > budget) {
          continue;
        }

        const int remaining_budget = budget - module_cost;
        if (probability_aggregates_built[remaining_budget] == 0) {
          build_aggregates_for_budget(remaining_budget, probability_table, probability_aggregates_cache[remaining_budget]);
          probability_aggregates_built[remaining_budget] = 1;
        }
        if (lock_key_aggregates_built[remaining_budget] == 0) {
          build_aggregates_for_budget(remaining_budget, expected_lock_key_table, lock_key_aggregates_cache[remaining_budget]);
          lock_key_aggregates_built[remaining_budget] = 1;
        }

        const double probability_mass = option_candidate_probability_masses_by_state[action_index];
        const double next_probability =
          get_aggregated_value_sum(
            protected_mask,
            state_index,
            single_compatibility_keys,
            pair_compatibility_keys,
            probability_aggregates_cache[remaining_budget][next_module_mask]
          ) /
          probability_mass;
        if (next_probability <= COMPARISON_EPSILON) {
          continue;
        }

        const double next_expected_lock_key_cost =
          action_lock_key_cost_by_mask_triplet[action_cost_index] +
          get_aggregated_value_sum(
            protected_mask,
            state_index,
            single_compatibility_keys,
            pair_compatibility_keys,
            lock_key_aggregates_cache[remaining_budget][next_module_mask]
          ) /
            probability_mass;

        if (is_better_candidate(next_probability, next_expected_lock_key_cost, best_probability, best_expected_lock_key_cost)) {
          best_probability = next_probability;
          best_expected_lock_key_cost = next_expected_lock_key_cost;
          best_action_type = 1;
          best_module_mask = next_module_mask;
          best_key_mask = key_mask;
        }
      }

      for (int candidate_index = 0; candidate_index < grade_candidate_counts_by_state[state_index]; ++candidate_index) {
        const int action_index = candidate_offset + candidate_index;
        const int next_module_mask = grade_candidate_next_module_masks_by_state[action_index];
        const int protected_mask = grade_candidate_protected_masks_by_state[action_index];
        const int key_mask = grade_candidate_key_masks_by_state[action_index];
        const int action_cost_index = build_action_cost_index(current_module_mask, next_module_mask, key_mask);
        const int module_cost = action_module_cost_by_mask_triplet[action_cost_index];
        if (module_cost > budget) {
          continue;
        }

        const int remaining_budget = budget - module_cost;
        const int grade_transition_count = grade_transition_counts_by_state_and_mask[state_index * MASK_COUNT + protected_mask];
        if (grade_transition_count == 0) {
          continue;
        }

        const int transition_offset = (state_index * MASK_COUNT + protected_mask) * MAX_GRADE_TRANSITIONS_PER_MASK;
        const int next_table_offset = remaining_budget * state_count;
        double next_probability = 0.0;
        double next_expected_lock_key_cost = action_lock_key_cost_by_mask_triplet[action_cost_index];

        for (int transition_index = 0; transition_index < grade_transition_count; ++transition_index) {
          const double probability = grade_transition_probabilities_by_state_and_mask[transition_offset + transition_index];
          const int grade_mask = grade_transition_grade_masks_by_state_and_mask[transition_offset + transition_index];
          const int next_state_index =
            next_state_index_by_module_mask_and_grade_mask[
              state_index * MASK_COUNT * GRADE_MASK_COUNT + next_module_mask * GRADE_MASK_COUNT + grade_mask
            ];
          next_probability += probability * probability_table[next_table_offset + next_state_index];
          next_expected_lock_key_cost += probability * expected_lock_key_table[next_table_offset + next_state_index];
        }

        if (next_probability <= COMPARISON_EPSILON) {
          continue;
        }

        if (is_better_candidate(next_probability, next_expected_lock_key_cost, best_probability, best_expected_lock_key_cost)) {
          best_probability = next_probability;
          best_expected_lock_key_cost = next_expected_lock_key_cost;
          best_action_type = 2;
          best_module_mask = next_module_mask;
          best_key_mask = key_mask;
        }
      }

      probability_table[table_index] = best_probability;
      expected_lock_key_table[table_index] = best_action_type == -1 ? 0.0 : best_expected_lock_key_cost;
      action_type_table[table_index] = static_cast<int8_t>(best_action_type);
      action_module_mask_table[table_index] = static_cast<int8_t>(best_module_mask);
      action_key_mask_table[table_index] = static_cast<int8_t>(best_key_mask);
    }
  }
}

}