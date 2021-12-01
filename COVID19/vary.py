import chaospy as cp

#########################
# parameters to vary    #
# place types           #
# ----------------------#
# 0 = elementary school #
# 1 = high school       #
# 2 = university        #
# 3 = workplaces        #
#########################

vary = {
    ###########################
    # Intervention parameters #
    ###########################
    "Relative_household_contact_rate_after_closure": cp.Uniform(1.5*0.8, 1.5*1.2),
    "Relative_spatial_contact_rate_after_closure": cp.Uniform(1.25*0.8, 1.25*1.2),
    "Relative_household_contact_rate_after_quarantine": cp.Uniform(2.0*0.8, 2.0*1.2),
    "Residual_spatial_contacts_after_household_quarantine": cp.Uniform(0.25*0.8, 0.25*1.2),
    "Household_level_compliance_with_quarantine": cp.Uniform(0.5, 0.9),
    "Individual_level_compliance_with_quarantine": cp.Uniform(0.9, 1.0),
    "Proportion_of_detected_cases_isolated":cp.Uniform(0.6, 0.8),
    "Residual_contacts_after_case_isolation":cp.Uniform(0.25*0.8, 0.25*1.2),
    "Relative_household_contact_rate_given_social_distancing":cp.Uniform(1.1, 1.25*1.2),
    "Relative_spatial_contact_rate_given_social_distancing":cp.Uniform(0.15, 0.35),
    # "Delay_to_start_household_quarantine":cp.DiscreteUniform(1, 3),
    # "Length_of_time_households_are_quarantined":cp.DiscreteUniform(12, 16),
    # "Delay_to_start_case_isolation":cp.DiscreteUniform(1, 3),
    # "Duration_of_case_isolation":cp.DiscreteUniform(5, 9),
    ######################
    # Disease parameters #
    ######################
    "Symptomatic_infectiousness_relative_to_asymptomatic": cp.Uniform(1,2),
    "Proportion_symptomatic": cp.Uniform(0.4,0.8),
    "Latent_period": cp.Uniform(3,6),
    "Mortality_factor": cp.Uniform(0.8,1.2),
    # "Reproduction_number": cp.Uniform(2,2.7),
    # "Infectious_period": cp.Uniform(11.5, 15.6),
    "Household_attack_rate": cp.Uniform(0.1, 0.19),
    "Household_transmission_denominator_power": cp.Uniform(0.7, 0.9),
    "Delay_from_end_of_latent_period_to_start_of_symptoms": cp.Uniform(0, 1.5),
    "Relative_transmission_rates_for_place_types0": cp.Uniform(0.08, 0.15),
    "Relative_transmission_rates_for_place_types1": cp.Uniform(0.08, 0.15),
    "Relative_transmission_rates_for_place_types2": cp.Uniform(0.05, 0.1),
    "Relative_transmission_rates_for_place_types3": cp.Uniform(0.05, 0.07),
    "Relative_spatial_contact_rates_by_age_power": cp.Uniform(0.25, 4),
    ######################
    # Spatial parameters #
    ######################
    "Proportion_of_places_remaining_open_after_closure_by_place_type2": cp.Uniform(0.2, 0.3),
    "Proportion_of_places_remaining_open_after_closure_by_place_type3": cp.Uniform(0.8, 1.0),
    "Residual_place_contacts_after_household_quarantine_by_place_type0": cp.Uniform(0.2, 0.3),
    "Residual_place_contacts_after_household_quarantine_by_place_type1": cp.Uniform(0.2, 0.3),
    "Residual_place_contacts_after_household_quarantine_by_place_type2": cp.Uniform(0.2, 0.3),
    "Residual_place_contacts_after_household_quarantine_by_place_type3": cp.Uniform(0.2, 0.3),
    "Relative_place_contact_rate_given_social_distancing_by_place_type0": cp.Uniform(0.8, 1.0),
    "Relative_place_contact_rate_given_social_distancing_by_place_type1": cp.Uniform(0.8, 1.0),
    "Relative_place_contact_rate_given_social_distancing_by_place_type2": cp.Uniform(0.6, 0.9),
    "Relative_place_contact_rate_given_social_distancing_by_place_type3": cp.Uniform(0.6, 0.9),
    "Relative_rate_of_random_contacts_if_symptomatic": cp.Uniform(0.4, 0.6),
    "Relative_level_of_place_attendance_if_symptomatic0": cp.Uniform(0.2, 0.3),
    "Relative_level_of_place_attendance_if_symptomatic1": cp.Uniform(0.2, 0.3),
    "Relative_level_of_place_attendance_if_symptomatic2": cp.Uniform(0.4, 0.6),
    "Relative_level_of_place_attendance_if_symptomatic3": cp.Uniform(0.4, 0.6),
    # "CLP1": cp.Uniform(60, 400)
    #############
    # Leftovers #
    #############
    "Kernel_scale": cp.Uniform(0.9*4000, 1.1*4000),
    "Kernel_Shape": cp.Uniform(0.8*3, 1.2*3),
    "Kernel_shape_params_for_place_types0": cp.Uniform(0.8*3, 1.2*3),
    "Kernel_shape_params_for_place_types1": cp.Uniform(0.8*3, 1.2*3),
    "Kernel_shape_params_for_place_types2": cp.Uniform(0.8*3, 1.2*3),
    "Kernel_shape_params_for_place_types3": cp.Uniform(0.8*3, 1.2*3),
    "Kernel_scale_params_for_place_types0": cp.Uniform(0.9*4000, 1.1*4000),
    "Kernel_scale_params_for_place_types1": cp.Uniform(0.9*4000, 1.1*4000),
    "Kernel_scale_params_for_place_types2": cp.Uniform(0.9*4000, 1.1*4000),
    "Kernel_scale_params_for_place_types3": cp.Uniform(0.9*4000, 1.1*4000),
    # "Param_1_of_place_group_size_distribution0": cp.DiscreteUniform(20, 30),
    # "Param_1_of_place_group_size_distribution1": cp.DiscreteUniform(20, 30),
    # "Param_1_of_place_group_size_distribution2": cp.DiscreteUniform(80, 120),
    # "Param_1_of_place_group_size_distribution3": cp.DiscreteUniform(8, 12),
    "Proportion_of_between_group_place_links0": cp.Uniform(0.8*0.25, 1.2*0.25),
    "Proportion_of_between_group_place_links1": cp.Uniform(0.8*0.25, 1.2*0.25),
    "Proportion_of_between_group_place_links2": cp.Uniform(0.8*0.25, 1.2*0.25),
    "Proportion_of_between_group_place_links3": cp.Uniform(0.8*0.25, 1.2*0.25),
}