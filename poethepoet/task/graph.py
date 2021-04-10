from typing import Dict, Set, List, Tuple
from ..config import PoeConfig
from ..exceptions import CyclicDependencyError
from .base import PoeTask

# TODO: better to have a separate option for no-capture???, deps/uses
NO_CAPTURE_KEY = ""


# TODO: have a graph object node objects instead of this sink relationship??
# TODO: executing a task with no deps should skip all this!


class TaskExecutionNode:
    task: PoeTask
    direct_dependants: List["TaskExecutionNode"]
    direct_dependencies: Set[str]
    path_dependants: Tuple[str, ...]

    def __init__(
        self,
        task: PoeTask,
        direct_dependants: List["TaskExecutionNode"],
        path_dependants: Tuple[str, ...],
    ):
        self.task = task
        self.direct_dependants = direct_dependants
        self.direct_dependencies = task.get_deps()
        self.path_dependants = (task.name, *path_dependants)

    def is_source(self):
        return not self.task.has_deps()


class TaskExecutionGraph:
    """
    A directed-acyclic execution graph of tasks, with a single sink node, and any number
    of source nodes. Non-source nodes may have multiple upstream nodes, and non-sink
    nodes may have multiple downstream nodes.

    A task/node may appear twice in the graph, if one instance has captured output, and
    one does not. Nodes are deduplicated to enforce this.
    """

    config: PoeConfig
    sources: List[TaskExecutionNode]
    sink: TaskExecutionNode
    captured_tasks: Dict[str, TaskExecutionNode]
    uncaptured_tasks: Dict[str, TaskExecutionNode]

    def __init__(
        self, sink_task: PoeTask, config: PoeConfig,
    ):
        self.config = config
        self.sink = TaskExecutionNode(sink_task, [], tuple())
        self.sources = []
        self.captured_tasks = {}
        self.uncaptured_tasks = {}

        # Build graph
        self._resolve_node_deps(self.sink)

    def get_execution_plan(self) -> List[List[PoeTask]]:
        """
        Derive an execution plan from the DAG in terms of stages consisting of tasks
        that could theoretically be parallelized.
        """
        stages = [self.sources]
        visited = set(source.task.name for source in self.sources)

        while True:
            next_stage = []
            for node in stages[-1]:
                for dep_node in node.direct_dependants:
                    if not dep_node.direct_dependencies.issubset(visited):
                        # Some dependencies of dep_node have not been visited, so skip them
                        # for now
                        continue
                    next_stage.append(dep_node)
            if next_stage:
                stages.append(next_stage)
                visited.update(node.task.name for node in next_stage)
            else:
                break

        return [[node.task for node in stage] for stage in stages]

    def _resolve_node_deps(self, node: TaskExecutionNode):
        """
        Build a DAG of tasks by depth-first traversal of the dependency tree starting
        from the sink node.
        """
        for key, deps in node.task.get_upstream_from_config(self.config).items():
            for task in deps:
                if task.name in node.path_dependants:
                    raise CyclicDependencyError(
                        f"Encountered cyclic task dependency at {task.name}"
                    )

                # Check if a node already exists for this task
                if key == NO_CAPTURE_KEY:
                    if task.name in self.uncaptured_tasks:
                        # reuse instance of task with uncaptured output
                        self.uncaptured_tasks[task.name].direct_dependants.append(node)
                        continue
                elif task.name in self.captured_tasks:
                    # reuse instance of task with captured output
                    self.captured_tasks[task.name].direct_dependants.append(node)
                    continue

                # This task has not been encountered before by another path
                new_node = TaskExecutionNode(task, [node], node.path_dependants)

                # Keep track of this task/node so it can be found by other dependants
                if key == NO_CAPTURE_KEY:
                    self.uncaptured_tasks[task.name] = new_node
                else:
                    self.captured_tasks[task.name] = new_node

                if new_node.is_source():
                    # Track this node as having no dependencies
                    self.sources.append(new_node)
                else:
                    # Recurse immediately for DFS
                    self._resolve_node_deps(new_node)
