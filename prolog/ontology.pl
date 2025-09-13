disease(gout).
disease(rheumatoid_arthritis).
abbreviation(rheumatoid_arthritis, ra).
cause(gout, uric_acid_crystals).
cause(rheumatoid_arthritis, autoimmune).
onset(gout, sudden_hours).
onset(rheumatoid_arthritis, gradual_weeks_months).
pain_peak(gout, first_24_hours).
pain_peak(rheumatoid_arthritis, builds_over_time).
joint_pattern(gout, monoarticular).
joint_pattern(rheumatoid_arthritis, symmetrical_polyarticular).
common_joint(gout, big_toe).
common_joint(gout, ankle).
common_joint(gout, knee).
common_joint(rheumatoid_arthritis, hands).
common_joint(rheumatoid_arthritis, wrists).
common_joint(rheumatoid_arthritis, feet).
morning_stiffness(gout, minimal_or_none).
morning_stiffness(rheumatoid_arthritis, over_1_hour).
systemic_symptoms_frequency(gout, rare_unless_severe).
systemic_symptom(rheumatoid_arthritis, fatigue).
systemic_symptom(rheumatoid_arthritis, fever).
systemic_symptom(rheumatoid_arthritis, malaise).
lesion_type(gout, tophi).
lesion_characteristic(gout, chalky_under_skin).
lesion_type(rheumatoid_arthritis, rheumatoid_nodules).
lesion_consistency(rheumatoid_arthritis, firm).
lesion_common_site(rheumatoid_arthritis, elbows).
blood_test(gout, high_uric_acid).
blood_test_variability(gout, high_uric_acid, not_always).
blood_test(rheumatoid_arthritis, rf_positive).
blood_test(rheumatoid_arthritis, anti_ccp_positive).
imaging(gout, bone_erosion_with_overhanging_edge).
imaging(rheumatoid_arthritis, symmetric_joint_space_narrowing).
imaging(rheumatoid_arthritis, erosions).
small_joint(hands).
small_joint(wrists).
small_joint(feet).
small_joint_involvement(D) :- common_joint(D, J), small_joint(J).
monoarticular(D) :- joint_pattern(D, often_monoarticular).
polyarticular(D) :- joint_pattern(D, symmetrical_polyarticular).
symmetric_involvement(D) :- joint_pattern(D, symmetrical_polyarticular).
rapid_onset(D) :- onset(D, sudden_hours).
gradual_onset(D) :- onset(D, gradual_weeks_months).
rapid_pain_peak(D) :- pain_peak(D, first_24_hours).
morning_stiffness_prolonged(D) :- morning_stiffness(D, over_1_hour).
autoimmune_disease(D) :- cause(D, autoimmune).
has_feature(D, cause, V) :- cause(D, V).
has_feature(D, onset, V) :- onset(D, V).
has_feature(D, pain_peak, V) :- pain_peak(D, V).
has_feature(D, joint_pattern, V) :- joint_pattern(D, V).
has_feature(D, common_joint, J) :- common_joint(D, J).
has_feature(D, morning_stiffness, V) :- morning_stiffness(D, V).
has_feature(D, systemic_symptom, S) :- systemic_symptom(D, S).
has_feature(D, systemic_symptoms_frequency, V) :- systemic_symptoms_frequency(D, V).
has_feature(D, lesion_type, L) :- lesion_type(D, L).
has_feature(D, lesion_characteristic, C) :- lesion_characteristic(D, C).
has_feature(D, lesion_consistency, C) :- lesion_consistency(D, C).
has_feature(D, lesion_common_site, S) :- lesion_common_site(D, S).
has_feature(D, blood_test, T) :- blood_test(D, T).
has_feature(D, imaging, I) :- imaging(D, I).