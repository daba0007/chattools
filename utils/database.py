from typing import List, Optional, Dict
from py2neo import Graph, Node, Relationship


class Neo4jDatabase:
    def __init__(self, host: str = "bolt://localhost:7687",
                 user: str = "neo4j",
                 password: str = "pleaseletmein"):
        """Initialize the movie database"""

        self.graph = Graph(host, auth=(user, password))
        self.limit = 10000  # 添加 limit 属性

    def query(
        self,
        cypher_query: str,
        params: Optional[Dict] = {}
    ) -> List[Dict[str, str]]:
        print(cypher_query)
        result = self.graph.run(cypher_query, parameters=params)
        return [dict(record) for record in result][:50]
    
    def get_g6_graph(self, cypher_query: str) -> Dict[str, List[Dict[str, str]]]:
        data = {"nodes": [], "edges": []}
        with self.graph.begin() as tx:
            result = tx.run(cypher_query, limit=self.limit)
            nodes = set()
            rels = set()

            for record in result:
                for key in record.keys():
                    if isinstance(record[key], Node):
                        nodes.add(record[key])
                    elif isinstance(record[key], Relationship):
                        rels.add(record[key])

            for node in nodes:
                node_properties = dict(node.items())
                node_properties['id'] = str(node.identity)
                node_properties['label'] = list(node.labels)[0]
                data['nodes'].append(node_properties)

            for rel in rels:
                rel_properties = dict(rel.items())
                rel_properties['source'] = str(rel.start_node.identity)
                rel_properties['target'] = str(rel.end_node.identity)
                rel_properties['label'] = list(rel.types())[0]
                data['edges'].append(rel_properties)

        return data