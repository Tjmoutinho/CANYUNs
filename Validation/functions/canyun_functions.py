import multiprocessing
import logging
import optlang
from warnings import warn
from itertools import product
from functools import partial
from builtins import (map, dict)
import symengine
import cobra
from cobra.core import *
from cobra.io import load_json_model
import copy
from copy import deepcopy
import pickle
import pandas as pd
import glob
import json
from optlang.symbolics import Zero
from cobra.core import Reaction
import sys
import operator
from cobra.util.util import format_long_string
import numpy as np
import random

from cobra.manipulation.delete import find_gene_knockout_reactions
import cobra.util.solver as sutil
from cobra.flux_analysis.moma import add_moma
from cobra.flux_analysis.room import add_room

LOGGER = logging.getLogger(__name__)

def convert_to_weight_space(model, annotation_evidence, objective_rxn_id='Biomass', 
                            compression = 0.0, threshold=500, no_ev_value=-1.0, pfba=False):
    """
    Convert the Annotation evidence from Probannopy to weight space.
    
    Parameters
    ----------
    direction : string
        'Forward' or 'Reverse'
    model : cobra.Model
        The metabolic model (universal).
    annotation_evidence : iterable, optional
        ``cobra.Reaction``s to be deleted. If not passed,
        all the reactions from the model are used.
    objective_rxn_id : string
        ID for objective function.
    compression : float
        Value between 0 and 1 representing value of compression for genetic annotation evidence.
    threshold : int
        Value for midpoint of annotation evidence values that will equal zero in weight space. 
    no_ev_value : float
        Weight space value for reactions without annotation evidence.
    pfba : True or False
        Parameter to convert all reaction weights to -1.0, thus performing pFBA when set to True. 

    Returns
    -------
    anno_ev_all : dictionary
        Annotation evidence values in weight space for all reactions. Rxn IDs are keys of dict. 
    """
    
    annotation_evidence_temp = deepcopy(annotation_evidence)
    for key, value in annotation_evidence_temp.items():
        if value[0] >= 1000:
            annotation_evidence_temp[key] = [1000.0]
    
    aemax = max([vals for vals in annotation_evidence_temp.values()])[0]
    
    if pfba == True:
        anno_ev_all = deepcopy(annotation_evidence_temp)
        for rxn in model.reactions:
            if not rxn.id.startswith('EX_') and not rxn.id.startswith('SNK_'):
                anno_ev_all[rxn.id] = [no_ev_value, no_ev_value]

    elif pfba == False:
        # Make annotation evidence dictionary with values for all reactions
        anno_ev_all = deepcopy(annotation_evidence_temp)
        for rxn in model.reactions:
            if not rxn.id.startswith('EX_') and not rxn.id.startswith('SNK_'):
                try:
                    if annotation_evidence_temp[rxn.id][0]>=threshold: # ((1-c)/(xmax-t))*(xi-t)+(c)
#                         print('check 1')
                        if rxn.lower_bound == -1000.0 and rxn.upper_bound == 1000.0:
                            # Reversible 
                            anno_ev_all[rxn.id] = \
                                [(((1-compression)/(aemax-threshold))*(annotation_evidence_temp[rxn.id][0] - threshold))+compression,
                                 (((1-compression)/(aemax-threshold))*(annotation_evidence_temp[rxn.id][0] - threshold))+compression]
                        elif rxn.lower_bound == 0.0 and rxn.upper_bound == 1000.0:
                            # Forward
                            anno_ev_all[rxn.id] = \
                                        [(((1-compression)/(aemax-threshold))*(annotation_evidence_temp[rxn.id][0] - threshold))+compression,
                                         no_ev_value]
                        elif rxn.lower_bound == -1000.0 and rxn.upper_bound == 0.0:
                            # Reverse
                            anno_ev_all[rxn.id] = \
                                        [no_ev_value,
                                         (((1-compression)/(aemax-threshold))*(annotation_evidence_temp[rxn.id][0] - threshold))+compression]
                        elif rxn.lower_bound == 0.0 and rxn.upper_bound == 0.0:
                             anno_ev_all[rxn.id] = \
                                        [no_ev_value, no_ev_value]

                    else: #If less than threshold  ((1-c)/t)*(xi)-1
                        if rxn.lower_bound == -1000.0 and rxn.upper_bound == 1000.0:
                            # Both
                            anno_ev_all[rxn.id] = \
                                [((1-compression)/threshold)*annotation_evidence_temp[rxn.id][0] - 1,
                                 ((1-compression)/threshold)*annotation_evidence_temp[rxn.id][0] - 1]
                        elif rxn.lower_bound == 0.0 and rxn.upper_bound == 1000.0:
                            # Forward
                            anno_ev_all[rxn.id] = \
                                [((1-compression)/threshold)*annotation_evidence_temp[rxn.id][0] - 1,
                                 no_ev_value]
                        elif rxn.lower_bound == -1000.0 and rxn.upper_bound == 0.0:
                            # Reverse
                            anno_ev_all[rxn.id] = \
                                [no_ev_value,
                                 ((1-compression)/threshold)*annotation_evidence_temp[rxn.id][0] - 1]
                        elif rxn.lower_bound == 0.0 and rxn.upper_bound == 0.0:
                             anno_ev_all[rxn.id] = \
                                        [no_ev_value, no_ev_value]
                except:
                    anno_ev_all[rxn.id] = [no_ev_value, no_ev_value]
            else: # No need for annotation evidence for EX or SNK reactions. Will not be used in dgFBA later. 
                continue

        anno_ev_all[objective_rxn_id] = [1.0, no_ev_value] # Biomass must be produced
                                                   # The biomass reaction can only go in the forward direction
    else:
        print('pfba argument must be True of False')
    
    return anno_ev_all

def write_dgFBA_expr(model, weights_dict):
    
    """
    Write dgFBA expression for running optimization.
    
    Parameters
    ----------
    model : cobra.Model
        The metabolic model (universal).
    weights_dict : dictionary
        Dictionary of reaction weights

    Returns
    -------
    dgfba_expr : expression
        Expression for optimization
    """
    
    # Set the pFBA objective if the model can grow
    dgfba_expr = Zero
    for rxn in model.reactions:
        if not rxn.id.startswith('EX_') and not rxn.id.startswith('SNK_'):
            dgfba_expr += weights_dict[rxn.id][0] * rxn.forward_variable
            dgfba_expr += weights_dict[rxn.id][1] * rxn.reverse_variable
    return dgfba_expr

def dgFBA(model, dgfba_expr, objective_rxn_id, percent_opt=0.1):
    
    """
    Run dgFBA to identify reactions that are required to carry flux.
    
    Parameters
    ----------
    model : cobra.Model
        The metabolic model (universal).
    dgfba_expr : expression
        Expression for optimization
    objective_rxn_id : string
        ID for objective function.
    percent_opt : float
        Percent of optimum objective flux value required. 

    Returns
    -------
    solution : cobra.Model solution data
        reaction flux values
    """
    # Set max biomass as objective to establish growth or no growth in condition
    BM_rxn = model.reactions.get_by_id(objective_rxn_id)
    BM_rxn.lower_bound = 0.0
    BM_rxn.upper_bound = 1000.0

    fba_expr = BM_rxn.forward_variable + BM_rxn.reverse_variable
    model.objective = model.problem.Objective(fba_expr, direction='max', sloppy=True)
    
    model.solver.update()
    
    fba_solution = model.optimize()
    fba_opt = fba_solution.objective_value
    

    # If there is growth, run dgFBA on the model
    if fba_opt > 0.1: # Growth threshold

        # Set X% of previous optimal BM value as constraint of model for pFBA, to force BM production
        BM_rxn.lower_bound = fba_opt*percent_opt
        BM_rxn.upper_bound = 1000.0

        # Run dgFBA and save solution
        model.objective = model.problem.Objective(dgfba_expr, direction='max', sloppy=True)
        model.solver.update()
        solution = model.optimize()
        return solution
    else:
        return fba_solution

def set_dgFBA_media(universal, media_list):
    
    """
    Run dgFBA to identify reactions that are required to carry flux.
    
    Parameters
    ----------
    universal : cobra.Model
        The universal metabolic network model
    media_list : list
        List of media compounds

    Returns
    -------
    Sets media condition for model object.
    """
    # Set all sink reactions to zero
    for rxn in universal.reactions:
        if rxn.id.startswith('SNK_'):
            rxn.upper_bound = 0.0
            rxn.lower_bound = 0.0
    # Set all exchanges to zero flux in and full flux out
    for rxn in universal.reactions:
        if rxn.id.startswith('EX_'):
            rxn.lower_bound = 0.0
            rxn.upper_bound = 1000.0
    # Open base media exchanges
    for met in media_list:
        try:
            # Search for exchange reactions
            temp_exchange = universal.reactions.get_by_id('EX_'+ met)
            temp_exchange.lower_bound = -1000.0
            temp_exchange.upper_bound = 1000.0
        except:
            pass

def make_canyun(model, dgfba_expr, well_id_media_dict, objective_rxn_id = 'Growth'):
    
    """
    Make all data objects for a CANYUN model
    
    Parameters
    ----------
    model : cobra.Model
        The metabolic model (universal).
    dgfba_expr : expression
        Expression for optimization
    well_id_media_dict : dictionary
        dictionary of media conditions; media compounds represented as lists of metabolite IDs
    objective_rxn_id : string
        ID for objective function.

    Returns
    -------
    output : list of objects
        Six different objects are created. Each object contains information about the flux values 
        for each solution across all of the media conditions. 
    """
    no_comp_growth = []
    comp_growth = []
    infeasible_list = []
    error_list = []
    active_rxn_dict = dict()
    solution_dict = dict()
    
    for key, media_list in well_id_media_dict.items(): 
        set_dgFBA_media(model, media_list)
        solution = dgFBA(model, dgfba_expr, objective_rxn_id)
        try:
            if abs(solution.objective_value) > 0.01:
                active_rxns = set([rxn.id for rxn in model.reactions if abs(solution.fluxes[rxn.id]) > 1e-6])
                comp_growth.append(key)
                active_rxn_dict[key] = active_rxns
                solution_dict[key] = solution

            elif abs(solution.objective_value) < 0.01 and solution.status != 'infeasible':
                no_comp_growth.append(key)

            elif solution.status == 'infeasible':
                infeasible_list.append(key)

        except:
            error_list.append(key)
            print(key, 'ERROR')

    output = [comp_growth, no_comp_growth, infeasible_list, error_list, solution_dict, active_rxn_dict]
    return output

def make_usage_dict(model, solution_dict, gng_dict_long):
    
    """
    Make a dictionary with data about which reactions are used across all growth conditions.
    
    Parameters
    ----------
    model : cobra.Model
        The metabolic model (universal).
    solution_dict : dictionary
        Dict with all solution information across computational growth conditions
    gng_dict_long : dictionary
        Dict with growth and no growth data for organism

    Returns
    -------
    usage_dict : dictionary
        Dict with ratio each reaction was used across all growth conditions. Key == rxn_id
    """
    usage_dict = {}
    count = 0
    for key, solution in solution_dict.items():
        if gng_dict_long[key] > 0.0:
            count += 1
            rxn_set = set([rxn.id for rxn in model.reactions if abs(solution.fluxes[rxn.id]) > 1e-6])
            for rxn_id in rxn_set:
                if rxn_id[0:3] not in ['EX_','SNK','Gro','sin','DM_','BIO','Bio']:

                    if solution.fluxes[rxn_id] > 1e-6: # Forward
                        try:
                            usage_dict[rxn_id] = [usage_dict[rxn_id][0] + 1, usage_dict[rxn_id][1]]
                        except:
                            usage_dict[rxn_id] = [1,0]
                    elif solution.fluxes[rxn_id] < -1e-6: # Reverse
                        try:
                            usage_dict[rxn_id] = [usage_dict[rxn_id][0], usage_dict[rxn_id][1] + 1]
                        except:
                            usage_dict[rxn_id] = [0,1]
                    else:
                        print('ERROR')

    # Adjust values to be ratios
    for key, value in usage_dict.items():
        usage_dict[key] = [value[0]/count, value[1]/count]
    return usage_dict

def add_exs(model, universal_model, media_conditions_dict):
    
    """"
    Add exchange reaction to a model from a universal model
    """"
    
    # Added all necessary exchanges to the gapfilled carveme model
    model_exs = deepcopy(model)

    met_id_list = []
    for media_list in media_conditions_dict.values():
        for met in media_list:
            # Search for exchange reactions
            try:
                temp_exchange = model.reactions.get_by_id('EX_'+ met)
            except:
                met_id_list.append(met)
    met_id_set = set(met_id_list)

    # Add missing exchanges to ML1515
    count = 0
    for met_id in met_id_set:
        try:
            ex_rxn = Reaction('EX_' + met_id)
            ex_rxn.name = "Exchange reaction for " + met_id
            ex_rxn.lower_bound = 0.0
            ex_rxn.upper_bound = 1000.0
            met = deepcopy(universal_model.metabolites.get_by_id(met_id))
            ex_rxn.add_metabolites({met:-1})
            model_exs.add_reactions([ex_rxn])
            count += 1
        except:
            print('ERROR:', met_id)
    print('Exchanges add:', count)
    return model_exs

def _reactions_knockouts_with_restore_r(model, reactions):
    """
    Adapted from COBRApy
    """
    with model:
        for reaction in reactions:
            reaction.lower_bound = 0.0
        growth = _get_growth(model)
    return [r.id for r in reactions], growth, model.solver.status

def _reactions_knockouts_with_restore_f(model, reactions):
    """
    Adapted from COBRApy
    """
    with model:
        for reaction in reactions:
            reaction.upper_bound = 0.0
        growth = _get_growth(model)
    return [r.id for r in reactions], growth, model.solver.status

def _get_growth(model):
    """
    Sourced from COBRApy
    """
    try:
        if 'moma_old_objective' in model.solver.variables:
            model.slim_optimize()
            growth = model.solver.variables.moma_old_objective.primal
        else:
            growth = model.slim_optimize()
    except optlang.exceptions.SolverError:
        growth = float('nan')
    return growth

def _reaction_deletion_r(model, ids):
    """
    Sourced from COBRApy
    """
    return _reactions_knockouts_with_restore_r(
        model,
        [model.reactions.get_by_id(r_id) for r_id in ids]
    )

def _reaction_deletion_f(model, ids):
    """
    Sourced from COBRApy
    """
    return _reactions_knockouts_with_restore_f(
        model,
        [model.reactions.get_by_id(r_id) for r_id in ids]
    )

def _reaction_deletion_worker_r(ids):
    """
    Sourced from COBRApy
    """
    global _model
    return _reaction_deletion_r(_model, ids)

def _reaction_deletion_worker_f(ids):
    """
    Sourced from COBRApy
    """
    global _model
    return _reaction_deletion_f(_model, ids)

def _gene_deletion_worker(ids):
    """
    Sourced from COBRApy
    """
    global _model
    return _gene_deletion(_model, ids)

def _init_worker(model):
    """
    Sourced from COBRApy
    """
    global _model
    _model = model

def _multi_deletion_fr(direction, model, entity, element_lists, method="fba",
                       solution=None, processes=None, **kwargs):
    """
    Adapted from COBRApy
    
    Provide a common interface for single or multiple knockouts.

    Parameters
    ----------
    direction : string
        'Forward' or 'Reverse'
    model : cobra.Model
        The metabolic model to perform deletions in.
    entity : 'gene' or 'reaction'
        The entity to knockout (``cobra.Gene`` or ``cobra.Reaction``).
    element_lists : list
        List of iterables ``cobra.Reaction``s or ``cobra.Gene``s (or their IDs)
        to be deleted.
    method: {"fba", "moma", "linear moma", "room", "linear room"}, optional
        Method used to predict the growth rate.
    solution : cobra.Solution, optional
        A previous solution to use as a reference for (linear) MOMA or ROOM.
    processes : int, optional
        The number of parallel processes to run. Can speed up the computations
        if the number of knockouts to perform is large. If not passed,
        will be set to the number of CPUs found.
    kwargs :
        Passed on to underlying simulation functions.

    Returns
    -------
    pandas.DataFrame
        A representation of all combinations of entity deletions. The
        columns are 'growth' and 'status', where

        index : frozenset([str])
            The gene or reaction identifiers that were knocked out.
        growth : float
            The growth rate of the adjusted model.
        status : str
            The solution's status.
    """
    solver = sutil.interface_to_str(model.problem.__name__)
    if method == "moma" and solver not in sutil.qp_solvers:
        raise RuntimeError(
            "Cannot use MOMA since '{}' is not QP-capable."
            "Please choose a different solver or use FBA only.".format(solver))

    if processes is None:
        try:
            processes = multiprocessing.cpu_count()
        except NotImplementedError:
            warn("Number of cores could not be detected - assuming 1.")
            processes = 1

    with model:
        if "moma" in method:
            add_moma(model, solution=solution, linear="linear" in method)
        elif "room" in method:
            add_room(model, solution=solution, linear="linear" in method,
                     **kwargs)

        args = set([frozenset(comb) for comb in product(*element_lists)])
        processes = min(processes, len(args))

        def extract_knockout_results(result_iter):
            result = pd.DataFrame([
                (frozenset(ids), growth, status)
                for (ids, growth, status) in result_iter
            ], columns=['ids', 'growth', 'status'])
            result.set_index('ids', inplace=True)
            return result

        if processes > 1:
            if direction == 'forward':
                worker = dict(gene=_gene_deletion_worker,
                              reaction=_reaction_deletion_worker_f)[entity]
            elif direction == 'reverse':
                worker = dict(gene=_gene_deletion_worker,
                              reaction=_reaction_deletion_worker_r)[entity]
            chunk_size = len(args) // processes
            pool = multiprocessing.Pool(
                processes, initializer=_init_worker, initargs=(model,)
            )
            results = extract_knockout_results(pool.imap_unordered(
                worker,
                args,
                chunksize=chunk_size
            ))
            pool.close()
            pool.join()
        else:
            if direction == 'forward':
                worker = dict(gene=_gene_deletion,
                              reaction=_reaction_deletion_f)[entity]
            elif direction == 'reverse':
                worker = dict(gene=_gene_deletion,
                              reaction=_reaction_deletion_r)[entity]
            results = extract_knockout_results(map(
                partial(worker, model), args))
        return results

def _entities_ids(entities):
    """
    Sourced from COBRApy
    """
    try:
        return [e.id for e in entities]
    except AttributeError:
        return list(entities)

def _element_lists(entities, *ids):
    """
    Sourced from COBRApy
    """
    lists = list(ids)
    if lists[0] is None:
        lists[0] = entities
    result = [_entities_ids(lists[0])]
    for l in lists[1:]:
        if l is None:
            result.append(result[-1])
        else:
            result.append(_entities_ids(l))
    return result

def single_reaction_deletion_fr(direction, model, reaction_list=None, method="fba",
                                solution=None, processes=None, **kwargs):
    """
    Adapted from COBRApy
    
    Knock out each reaction from a given list.

    Parameters
    ----------
    direction : string
        'Forward' or 'Reverse'
    model : cobra.Model
        The metabolic model to perform deletions in.
    reaction_list : iterable, optional
        ``cobra.Reaction``s to be deleted. If not passed,
        all the reactions from the model are used.
    method: {"fba", "moma", "linear moma", "room", "linear room"}, optional
        Method used to predict the growth rate.
    solution : cobra.Solution, optional
        A previous solution to use as a reference for (linear) MOMA or ROOM.
    processes : int, optional
        The number of parallel processes to run. Can speed up the computations
        if the number of knockouts to perform is large. If not passed,
        will be set to the number of CPUs found.
    kwargs :
        Keyword arguments are passed on to underlying simulation functions
        such as ``add_room``.

    Returns
    -------
    pandas.DataFrame
        A representation of all single reaction deletions. The columns are
        'growth' and 'status', where

        index : frozenset([str])
            The reaction identifier that was knocked out.
        growth : float
            The growth rate of the adjusted model.
        status : str
            The solution's status.

    """
    return _multi_deletion_fr(direction,
        model, 'reaction',
        element_lists=_element_lists(model.reactions, reaction_list), 
        method=method, solution=solution, processes=processes, **kwargs)

def extract_essential_rxn_ids(ess_rxns_df):
    
    """
    Extract all essential reactions from dataframe with each essential reaction per growth condition.

    Parameters
    ----------
    ess_rxns_df : Pandas dataframe of essential reactions across all growth conditions.

    Returns
    -------
    ess_rxn_list : a list of essential reactions

    """
    
    ess_rxn_list = []
    ess_rxns_df_2 = ess_rxns_df.loc[(ess_rxns_df['growth'] < 1e-6)]
    for rxn_set in ess_rxns_df_2.index.tolist():
        rxn_id = list(rxn_set)[0]
        if rxn_id[0:3] not in ['EX_','SNK','Gro','sin','DM_','BIO','Bio']:
            ess_rxn_list.append(rxn_id)
    return ess_rxn_list

def find_f_essential_rxns(model, key, well_id_media_dict):
    
    """
    Find essential reactions in forward direction and output list for a single growth condition.

    Parameters
    ----------
    model : cobra.Model
        CANYUNs pre-processed model
    key : string
        key for media condition dictionary
    well_id_media_dict : dictionary
        dictionary of media conditions; media compounds represented as lists of metabolite IDs

    Returns
    -------
    f_ess_rxn_list : a list of essential reactions in the forward direction for one media condition.
    
    """

    model_temp = deepcopy(model)

    media_list = well_id_media_dict[key]
    objective_rxn_id = 'Biomass'

    set_dgFBA_media(model_temp, media_list)
    essential_rxns = []
    semi_essential_rxns = []
    essential_rxns_r = []
    essential_rxns_f = []

    BM_rxn = model_temp.reactions.get_by_id(objective_rxn_id)
    BM_rxn.lower_bound = 0.0
    BM_rxn.upper_bound = 1000.0
    fba_expr = BM_rxn.forward_variable
    model_temp.objective = model_temp.problem.Objective(fba_expr, direction='max', sloppy=True)
    model_temp.solver.update()
    fba_optimum = model_temp.slim_optimize()
    print(key, fba_optimum)

    f_ess_rxns_df = single_reaction_deletion_fr('forward', model_temp)
    f_ess_rxn_list = extract_essential_rxn_ids(f_ess_rxns_df)
    
    return f_ess_rxn_list

def find_r_essential_rxns(model, key, well_id_media_dict):

    """
    Find essential reactions in forward direction and output list for a single growth condition.

    Parameters
    ----------
    model : cobra.Model
        CANYUNs pre-processed model
    key : string
        key for media condition dictionary
    well_id_media_dict : dictionary
        dictionary of media conditions; media compounds represented as lists of metabolite IDs

    Returns
    -------
    r_ess_rxn_list : a list of essential reactions in the forward direction for one media condition.
    
    """
    
    model_temp = deepcopy(model)

    media_list = well_id_media_dict[key]
    objective_rxn_id = 'Biomass'

    set_dgFBA_media(model_temp, media_list)
    essential_rxns = []
    semi_essential_rxns = []
    essential_rxns_r = []
    essential_rxns_f = []

    BM_rxn = model_temp.reactions.get_by_id(objective_rxn_id)
    BM_rxn.lower_bound = 0.0
    BM_rxn.upper_bound = 1000.0
    fba_expr = BM_rxn.forward_variable
    model_temp.objective = model_temp.problem.Objective(fba_expr, direction='max', sloppy=True)
    model_temp.solver.update()
    fba_optimum = model_temp.slim_optimize()
    print(key, fba_optimum)

    r_ess_rxns_df = single_reaction_deletion_fr('reverse', model_temp)
    r_ess_rxn_list = extract_essential_rxn_ids(r_ess_rxns_df)
    
    return r_ess_rxn_list