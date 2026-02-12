"""Simple script to visualize POMCP search tree structure.

This provides utility functions to:
1. Export tree to text representation
2. Export tree to JSON for external visualization
3. Print tree statistics
"""

import json
import math
from typing import Dict, Any, Optional
import posggym


def compute_detailed_stats(node, depth=0, is_root=True):
    """Compute detailed tree statistics."""
    stats = {
        'obs_nodes': 0,
        'action_nodes': 0,
        'leaf_nodes': 0,

        # Whether the search tree contains any absorbing (terminal) ObsNode.
        # In posggym_baselines, ObsNode has `is_absorbing` which is set when `ego_done=True`.
        'reached_terminal_state': False,

        'total_particles': 0,
        'total_visits': 0,

        'max_depth': depth,

        'max_obs_children': 0,
        'max_action_children': 0,

        'depth_histogram': {},
        
        # Root node specific stats
        'root_particles': 0,
        'root_visits': 0,
        'root_unique_states': {},
    }
    
    if hasattr(node, 'obs'):  # ObsNode - retrieve stats from children, which are ActionNodes
        stats['obs_nodes'] = 1
        stats['total_visits'] = node.visits
        stats['max_action_children'] = len(node.children)

        if getattr(node, 'is_absorbing', False):
            stats['reached_terminal_state'] = True
        
        stats['total_particles'] = node.belief.size() # (obviously) obs node has belief
        stats['depth_histogram'][depth] = stats['depth_histogram'].get(depth, 0) + 1 # depth is counted based on observation nodes
        
        # Compute root-specific stats if this is the root node
        if is_root:
            stats['root_particles'] = node.belief.size()
            stats['root_visits'] = node.visits
            
            # Analyze unique states in root belief
            unique_robot_states = {}
            unique_obj_coords = {}
            unique_obj_status = {}
            
            for particle in node.belief.particles:
                # Assuming state is (robot_coord, obj_coord, obj_status, ...)
                robot_state = particle.state[0]
                obj_coord = particle.state[1]
                obj_status = particle.state[2]
                
                unique_robot_states[robot_state] = unique_robot_states.get(robot_state, 0) + 1
                unique_obj_coords[obj_coord] = unique_obj_coords.get(obj_coord, 0) + 1
                unique_obj_status[obj_status] = unique_obj_status.get(obj_status, 0) + 1
            
            stats['root_unique_robot_states'] = unique_robot_states
            stats['root_unique_obj_coords'] = unique_obj_coords
            stats['root_unique_obj_status'] = unique_obj_status
        
        if len(node.children) == 0:  # recursion base case
            stats['leaf_nodes'] = 1
        
        for action_node in node.children.values():
            child_stats = compute_detailed_stats(action_node, depth + 1, is_root=False)
            for key in ['obs_nodes', 'action_nodes', 'total_particles', 'total_visits', 'leaf_nodes']:
                stats[key] += child_stats[key]
            
            stats['max_depth'] = max(stats['max_depth'], child_stats['max_depth'])
            stats['max_obs_children'] = max(stats['max_obs_children'], child_stats['max_obs_children'])
            stats['max_action_children'] = max(stats['max_action_children'], child_stats['max_action_children'])

            stats['reached_terminal_state'] = (
                stats['reached_terminal_state'] or child_stats['reached_terminal_state']
            )

            for d, count in child_stats['depth_histogram'].items():
                stats['depth_histogram'][d] = stats['depth_histogram'].get(d, 0) + count
    
    else:  # ActionNode - retrieve stats from children, which are ObsNodes
        stats['action_nodes'] = 1
        stats['total_visits'] = node.visits
        stats['max_obs_children'] = len(node.children)
        
        if len(node.children) == 0: # recursion base case
            stats['leaf_nodes'] = 1
        
        for obs_node in node.children.values():
            child_stats = compute_detailed_stats(obs_node, depth + 1, is_root=False)
            for key in ['obs_nodes', 'action_nodes', 'total_particles', 'total_visits', 'leaf_nodes']:
                stats[key] += child_stats[key]
            stats['max_depth'] = max(stats['max_depth'], child_stats['max_depth'])
            stats['max_obs_children'] = max(stats['max_obs_children'], child_stats['max_obs_children'])
            stats['max_action_children'] = max(stats['max_action_children'], child_stats['max_action_children'])

            stats['reached_terminal_state'] = (
                stats['reached_terminal_state'] or child_stats['reached_terminal_state']
            )

            for d, count in child_stats['depth_histogram'].items():
                stats['depth_histogram'][d] = stats['depth_histogram'].get(d, 0) + count
    
    return stats


def print_detailed_stats(root, 
                        #  step_stats
                         ):
    
    """Print comprehensive tree statistics."""
    stats = compute_detailed_stats(root)
    
    print("=" * 70)
    print("DETAILED TREE STATISTICS")
    print("=" * 70)

        #################################3
    print("\nRoot Node Statistics (where action selection happens):")
    root_particles = stats['root_particles']
    root_visits = stats['root_visits']
    
    print(f"  Particles: {root_particles}")
    print(f"  Visits: {root_visits}")
    
    # Print uniqueness statistics
    unique_robot_states = list(stats.get('root_unique_robot_states', {}).items())
    unique_obj_coords = list(stats.get('root_unique_obj_coords', {}).items())
    unique_obj_status = list(stats.get('root_unique_obj_status', {}).items())
    

    print(f"  Unique Particle set: ")
    # only show if there are 2 or more unique set
    unique_set = 2
    # Show unique particle  distribution
    if len(unique_robot_states) >= unique_set:
        sorted_states = sorted(unique_robot_states, key=lambda x: x[1], reverse=True)
        limit = 10 if len(sorted_states) > 10 else len(sorted_states)
        for state, count in sorted_states[:limit]:
            percentage = (count / root_particles) * 100
            print(f"    {state}: {count} particles ({percentage:.1f}%)")
        if len(sorted_states) > 10:
            print(f"    ... and {len(sorted_states) - 10} more")
    
    if len(unique_obj_coords) >= unique_set:
        sorted_coords = sorted(unique_obj_coords, key=lambda x: x[1], reverse=True)
        limit = 10 if len(sorted_coords) > 10 else len(sorted_coords)
        for coord, count in sorted_coords[:limit]:
            percentage = (count / root_particles) * 100
            print(f"    {coord}: {count} particles ({percentage:.1f}%)")
        if len(sorted_coords) > 10:
            print(f"    ... and {len(sorted_coords) - 10} more")
    
    if len(unique_obj_status) >= unique_set:
        sorted_status = sorted(unique_obj_status, key=lambda x: x[1], reverse=True)
        for status, count in sorted_status:
            percentage = (count / root_particles) * 100
            print(f"    {status}: {count} particles ({percentage:.1f}%)")

    ################################
    print("\nTree structure:")
    # print(f"  Max depth: {stats['max_depth']}")
    print(f"  Max action children: {stats['max_action_children']}")
    print(f"  Max obs children: {stats['max_obs_children']}")

    ################################
    # Tree-scan: True iff any ObsNode in the current tree is marked absorbing.
    reached_terminal = stats['reached_terminal_state']
    if reached_terminal:
        pass
    print("\nPlanning has reached terminal state (in simulations):")
    print(f"  {reached_terminal}")

    print("\nDepth histogram of Obs nodes:")
    for depth in sorted(stats['depth_histogram'].keys()):
        count = stats['depth_histogram'][depth]
        bar = "█" * round(min(count, 300) / 6)
        print(f"{depth:2d}: {count:4d} nodes {bar}")
    
    print("\nNode counts:")
    print(f"  Obs nodes: {stats['obs_nodes']}", end=" ||")
    print(f"  Action nodes: {stats['action_nodes']}", end=" ||")
    # print(f"  Total nodes: {stats['obs_nodes'] + stats['action_nodes']}")
    print(f"  Leaf nodes: {stats['leaf_nodes']}")

    print("\nTotal visits across all nodes: ", stats['total_visits'])
    print("Total particles across all Obs nodes: ", stats['total_particles'])
    print("Average particles per Obs node: ", stats['total_particles'] / max(stats['obs_nodes'], 1))


    # ################################
    # print("\nSearch Iteration statistics:")
    # for key, value in step_stats.items():
    #     # only print certain keys
    #     if key not in ['num_sims', 'search_depth', 
    #                    'min_value', 'max_value' # Q-value of actions
    #                    ]:
    #         continue

    #     if isinstance(value, float):
    #         print(f"  {key}: {value:.4f}")
    #     else:
    #         print(f"  {key}: {value}")


def print_action_ranking(root, planner_config):
    """Print actions ranked by value and visits."""
    print("\n" + "=" * 70)
    print("ACTION RANKING")
    print("=" * 70)
    
    print("ACTIONS_STR = [\"0\", \"U\", \"D\", \"L\", \"R\"]")
    if len(root.children) == 0:
        print("No actions explored yet.")
        return
    
    # Collect action info
    actions_info = []
    for action, action_node in root.children.items():
        actions_info.append({
            'action': action,
            'value': action_node.value,
            'visits': action_node.visits,
            'total_value': action_node.total_value,
            'variance': action_node.variance,
            'num_obs': len(action_node.children),
        })
    
    # only if ucb is used
    if planner_config.action_selection == "ucb":
        for info in actions_info:
            ucb = (info['value'] + 
                   planner_config.c * math.sqrt(math.log(root.visits + 1) / (info['visits'] + 1))) # +1 to ensure numberical stability
            info['ucb'] = ucb
        
        # Sort by UCB (consider V and N)
        print("\nRanked by UCB:")
        sorted_by_ucb = sorted(actions_info, key=lambda x: x['ucb'], reverse=True)
        for i, info in enumerate(sorted_by_ucb, 1):
            print(f"  {i}. Action {info['action']}: "
                  f"ucb={info['ucb']:.3f}, v={info['value']:.3f}, "
                  f"n={info['visits']}, σ²={info['variance']:.3f}, obs={info['num_obs']}")
        
        # # Sort by value

        # # Sort by visits


def print_tree_ascii(node, depth=0, prefix="", max_depth=None, is_best_path=False):
    """Print tree in ASCII format with branches, inspired by pomdp-py TreeDebugger.
    
    Args:
        node: POMCP tree node (ObsNode or ActionNode)
        depth: Current depth in tree
        prefix: String prefix for branches
        max_depth: Maximum depth to print (None for unlimited)
        is_best_path: If True, only show best-value children
    """
    if max_depth is not None and depth > max_depth:
        return
    
    # Format node info
    node_type = "ObsNode" if hasattr(node, 'obs') else "ActionNode"
    visits = node.visits
    value = f"{node.value:.3f}" if hasattr(node, 'value') else "N/A"
    
    if hasattr(node, 'obs'):
        node_info = f"{node_type}(obs={node.obs}, n={visits}, v={value})"
        particles = node.belief.size() if hasattr(node.belief, 'size') else 0
        node_info += f" [particles={particles}]"
    else:
        node_info = f"{node_type}(action={node.action}, n={visits}, v={value})"
    
    # Print current node
    if depth == 0:
        print(node_info)
    else:
        print(f"{prefix}{node_info}")
    
    # Handle children
    if len(node.children) == 0:
        return
    
    children_list = list(node.children.items())
    
    # If showing best path, only follow highest-value child
    if is_best_path:
        if len(children_list) > 0:
            best_child = max(children_list, key=lambda x: x[1].value)
            children_list = [best_child]
    
    for i, (edge, child) in enumerate(children_list):
        is_last = (i == len(children_list) - 1)
        
        # Create branch characters
        if is_last:
            branch = "└─── "
            child_prefix = prefix + "     "
        else:
            branch = "├─── "
            child_prefix = prefix + "│    "
        
        # Print edge label
        edge_str = str(edge)[:20]  # Limit edge string length
        print(f"{prefix}{branch}{edge_str}")
        
        # Recursively print child
        print_tree_ascii(child, depth + 1, child_prefix, max_depth, is_best_path)


def print_best_path(planner, max_depth=None):
    """Print the best-value path through the tree (highest value at each step).
    
    Args:
        planner: POMCP planner object
        max_depth: Maximum depth to traverse
    """
    print("\n" + "=" * 70)
    print("BEST-VALUE PATH THROUGH TREE")
    print("=" * 70)
    
    print_tree_ascii(planner.root, max_depth=max_depth, is_best_path=True)


def tree_to_dict(node, depth=0, max_depth: Optional[int] = None, parent_type: str = "root"):
    """Convert tree node to dictionary representation."""
    if max_depth is not None and depth > max_depth:
        return None
    
    node_dict = {
        'type': parent_type,
        'depth': depth,
        'visits': node.visits,
        'value': node.value if hasattr(node, 'value') else None,
        'total_value': node.total_value if hasattr(node, 'total_value') else None,
        'variance': node.variance if hasattr(node, 'variance') else None,
    }
    
    if hasattr(node, 'obs'):  # ObsNode
        node_dict['type'] = 'observation'
        node_dict['obs'] = str(node.obs)
        node_dict['particles'] = node.belief.size() if hasattr(node.belief, 'size') else 0
        node_dict['children'] = {}
        
        for action, action_node in node.children.items():
            child_dict = tree_to_dict(action_node, depth + 1, max_depth, "action")
            if child_dict is not None:
                node_dict['children'][str(action)] = child_dict
    
    else:  # ActionNode
        node_dict['type'] = 'action'
        node_dict['action'] = node.action if hasattr(node, 'action') else None
        node_dict['children'] = {}
        
        for obs, obs_node in node.children.items():
            child_dict = tree_to_dict(obs_node, depth + 1, max_depth, "observation")
            if child_dict is not None:
                node_dict['children'][str(obs)] = child_dict
    
    return node_dict


# def save_tree_to_json(planner: POMCP, filename: str, max_depth: Optional[int] = None):
#     """Export POMCP search tree to JSON file for external visualization.
    
#     Args:
#         planner: POMCP planner object
#         filename: Output JSON filename
#         max_depth: Maximum tree depth to include (None for full tree)
#     """
#     tree_dict = tree_to_dict(planner.root, max_depth=max_depth)
    
#     # Add metadata
#     output = {
#         'tree': tree_dict,
#         'metadata': {
#             'total_visits': planner.root.visits,
#             'max_depth': max_depth if max_depth is not None else 'unlimited',
#             'config': {
#                 'discount': planner.config.discount if hasattr(planner.config, 'discount') else None,
#                 'c': planner.config.c if hasattr(planner.config, 'c') else None,
#                 'action_selection': planner.config.action_selection if hasattr(planner.config, 'action_selection') else None,
#             }
#         }
#     }
    
#     with open(filename, 'w') as f:
#         json.dump(output, f, indent=2)
    
#     print(f"Tree exported to {filename}")
