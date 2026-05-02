# MUBEEN KHALID i240605
#Ashar Ahmed i240621

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.widgets import Slider, Button
from itertools import combinations
import os
import glob
from sklearn.metrics.pairwise import cosine_similarity

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 90
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 5

class CPINetworkAnalyzer:
    def __init__(self):
       
        self.product_category_mapping ={
            "wheat": "Food Staples & Grains","rice": "Food Staples & Grains","bread": "Food Staples & Grains","pulse": "Food Staples & Grains","sugar": "Food Staples & Grains",
            "gur": "Food Staples & Grains","salt": "Food Staples & Grains","chilly": "Food Staples & Grains","garlic": "Food Staples & Grains","tea": "Food Staples & Grains",
            "daal": "Food Staples & Grains","beef": "Meat, Poultry & Dairy","mutton": "Meat, Poultry & Dairy","chicken": "Meat, Poultry & Dairy","milk": "Meat, Poultry & Dairy",
            "curd": "Meat, Poultry & Dairy","eggs": "Meat, Poultry & Dairy","tea prepared": "Meat, Poultry & Dairy","mustard oil": "Oils, Condiments & Sweeteners",
            "cooking oil": "Oils, Condiments & Sweeteners","vegetable ghee": "Oils, Condiments & Sweeteners","bananas": "Fruits & Vegetables","potatoes": "Fruits & Vegetables",
            "onions": "Fruits & Vegetables","tomatoes": "Fruits & Vegetables","washing soap": "Non-Food Essentials","match box": "Non-Food Essentials","soap": "Non-Food Essentials",
            "energy saver": "Non-Food Essentials","firewood": "Non-Food Essentials","cigarettes": "Non-Food Essentials","elec charges": "Utilities & Transport",
            "gas charges": "Utilities & Transport","kerosene": "Utilities & Transport","petrol": "Utilities & Transport","diesel": "Utilities & Transport","lpg": "Utilities & Transport",
            "telephone": "Utilities & Transport","cng": "Utilities & Transport","long cloth": "Clothing & Miscellaneous","shirting": "Clothing & Miscellaneous",
            "lawn": "Clothing & Miscellaneous","georgette": "Clothing & Miscellaneous","sandal": "Clothing & Miscellaneous","chappal": "Clothing & Miscellaneous",
            "cement": "Clothing & Miscellaneous","carpenter": "Clothing & Miscellaneous","mason": "Clothing & Miscellaneous","labour": "Clothing & Miscellaneous",
            "plumber": "Clothing & Miscellaneous","electrician": "Clothing & Miscellaneous","urea": "Clothing & Miscellaneous","ammonium nitrate": "Clothing & Miscellaneous",
            "phosphate": "Clothing & Miscellaneous","potash": "Clothing & Miscellaneous","npk": "Clothing & Miscellaneous"
        }

        # ADD ALL KEYWORD_PATTERN ALIASES INTO THE SINGLE MAPPING (this eliminates the duplicate mapping)
        self.product_category_mapping.update({
            'flour': 'Food Staples & Grains',
            'oil': 'Oils, Condiments & Sweeteners',
            'ghee': 'Oils, Condiments & Sweeteners',
            'banana': 'Fruits & Vegetables',
            'potato': 'Fruits & Vegetables',
            'tomato': 'Fruits & Vegetables',
            'lifebuoy': 'Non-Food Essentials',
            'electricity': 'Utilities & Transport',
        })

        #CPI IMPORTANCE WEIGHTS Constant
        self.category_importance_weights = {
            'Food Staples & Grains': 0.25,
            'Meat, Poultry & Dairy': 0.15,
            'Oils, Condiments & Sweeteners': 0.10,
            'Fruits & Vegetables': 0.10,
            'Non-Food Essentials': 0.08,
            'Utilities & Transport': 0.20,
            'Clothing & Miscellaneous': 0.12,
        }
        
        self.categories = self._build_categories_from_mapping()
        
        self.data = None
        self.cities = []
        self.years = []
        self.months = []
        self.networks = {}
        self.centrality_results = {}
        self.similarity_threshold = 0.7
        self._build_enhanced_keyword_map()   # this now only builds full_norm_map and token_map only


    def _build_categories_from_mapping(self):
        category_dict = {cat: [] for cat in self.category_importance_weights.keys()}
        for product, category in self.product_category_mapping.items():
            if category in category_dict:
                category_dict[category].append(product)
        return category_dict
      

    def _normalize_city_key(self, city_name: str) -> str:
        if pd.isna(city_name):
            return ""
        s = str(city_name).strip()
        return s


    def map_product_to_category(self, product_name):
        if pd.isna(product_name):
            return 'Uncategorized'
        prod_orig = str(product_name).strip()
        prod_lower = prod_orig.lower()

        if prod_lower in self.full_norm_map:
            return self.full_norm_map[prod_lower]

        # substring match with longer keys first (already includes all aliases now)
        for norm_key in sorted(self.full_norm_map.keys(), key=lambda x: -len(x)):
            if norm_key and norm_key in prod_lower:
                return self.full_norm_map[norm_key]


        # token fallback (still useful for weird spellings)
        prod_tokens = set([t for t in prod_lower.split() if len(t) > 2])
        token_category_scores = {}
        for token, cat in self.token_map.items():
            if token in prod_tokens:
                token_category_scores[cat] = token_category_scores.get(cat, 0) + 1

        if token_category_scores:
            return max(token_category_scores, key=token_category_scores.get)

        return 'Uncategorized'


    def _build_enhanced_keyword_map(self):
        self.full_norm_map = {}
        self.token_map = {}

        for raw_keyword, cat in self.product_category_mapping.items():   
            norm = raw_keyword.lower().strip()
            if norm:
                self.full_norm_map[norm] = cat

                tokens = [t for t in norm.split() if len(t) > 2]
                for t in tokens:
                    if t not in self.token_map:         
                        self.token_map[t] = cat
                        
    def clean_city_name(self, city_name):
        if pd.isna(city_name):
            return 'Unknown'
        city = str(city_name).strip().title()
        city_key = self._normalize_city_key(city)
        city_mapping = {
            'Islam-abad': 'Islamabad', 'Rawal-pindi': 'Rawalpindi', 'Gujran-wala': 'Gujranwala',
            'Faisal-abad': 'Faisalabad', 'Sar-godha': 'Sargodha', 'Baha-walpur': 'Bahawalpur',
            'Hyder-abad': 'Hyderabad', 'Pesha-war': 'Peshawar', 'Khuz-dar': 'Khuzdar',
            'Multan': 'Multan', 'Lahore': 'Lahore', 'Karachi': 'Karachi',
        }
        return city_mapping.get(city_key, city)

    def load_data(self, file_paths):
        if isinstance(file_paths, str):
            file_paths = [file_paths]
        
        dfs = []
        
        for path in file_paths:
            if not os.path.exists(path):
                continue
                
            df = pd.read_csv(path).copy()
            df.columns = df.columns.str.strip()
            
            lc = {c.lower(): c for c in df.columns}
            
            p_col = next((lc[k] for k in ['description','descriptiol','item','product','commodity','average pr'] if k in lc), None)
            c_col = next((lc[k] for k in ['cities','city','location','region'] if k in lc), None)
            pr_col = next((lc[k] for k in ['value','price','average price','price rs','average'] if k in lc), None)
            d_col = next((lc[k] for k in ['month','date','period','time','date'] if k in lc), None)
            
            if None in (p_col, c_col, pr_col, d_col):
                continue
                
            df = df[[p_col, c_col, pr_col, d_col]].rename(columns={p_col:'Product', c_col:'City', pr_col:'Price', d_col:'Date'})
            
            # Remove rows with invalid critical values (exact same logic as before, just vectorised)
            bad = pd.Series([False]*len(df))
            for col in ['City', 'Price', 'Date']:
                s = df[col].astype(str).str.strip()
                bad |= df[col].isna() | (s == '') | s.isin(['---','N/A','NULL','nan','NaN'])
            df = df[~bad]
            
            if df.empty:
                continue
                
            df['City'] = df['City'].apply(self.clean_city_name)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date'])
            if df.empty:
                continue
                
            df['Year'] = df['Date'].dt.year.astype(int)
            df['Month'] = df['Date'].dt.month.astype(int)
            df['Category'] = df['Product'].apply(self.map_product_to_category)
            
            uncat = df['Category'] == 'Uncategorized'
            if uncat.any():
                df.loc[uncat, 'Category'] = df.loc[uncat, 'Product'].apply(self.map_product_to_category)
                
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
            df = df[df['Price'] > 0].dropna(subset=['Price'])
            
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            self.data = None
            return
        
        self.data = pd.concat(dfs, ignore_index=True)
        self.cities = sorted(self.data['City'].unique())
        self.years = sorted(self.data['Year'].unique())
        self.months = sorted(self.data['Month'].unique())
        
    def get_products_by_category(self, category_name):
        """Return list of products for a given category"""
        return [prod for prod, cat in self.product_category_mapping.items() if cat == category_name]

    def show_category_mapping_summary(self):
        if self.data is None:
            return
        cat_counts = self.data['Category'].value_counts()
        uncategorized = (self.data['Category'] == 'Uncategorized').sum() if 'Category' in self.data.columns else 0

    def normalize_prices(self, prices):
        prices = np.array(prices, dtype=float)
        if prices.size == 0:
            return prices
        min_price = np.min(prices)
        max_price = np.max(prices)
        if max_price == min_price:
            return np.ones_like(prices)
        return (prices - min_price) / (max_price - min_price)

    def compute_cosine_similarity(self, vec_i, vec_j):
        vec_i = np.array(vec_i, dtype=float)
        vec_j = np.array(vec_j, dtype=float)
        norm_i = np.linalg.norm(vec_i)
        norm_j = np.linalg.norm(vec_j)
        if norm_i == 0 or norm_j == 0:
            return 0.0
        similarity = cosine_similarity(vec_i.reshape(1, -1), vec_j.reshape(1, -1))[0][0]
        return max(0.0, min(1.0, float(similarity)))

    def construct_price_vectors(self, year, category_name):
        if self.data is None:
            return {}
        subset = self.data[(self.data['Year'] == year) & (self.data['Category'] == category_name)]
        if subset.shape[0] == 0:
            return {}
        unique_products = subset['Product'].unique()
        city_product_prices = subset.groupby(['City', 'Product'])['Price'].mean().reset_index()
        price_matrix = city_product_prices.pivot(index='City', columns='Product', values='Price').fillna(0)
        price_vectors = {}
        for city in price_matrix.index:
            prices = price_matrix.loc[city].values
            price_vectors[city] = self.normalize_prices(prices)
        cols_len = len(price_matrix.columns)
        for city in self.cities:
            if city not in price_vectors:
                price_vectors[city] = np.zeros(cols_len)
        return price_vectors

    def construct_similarity_relation(self, year, category_name, threshold=None):
        if threshold is None:
            threshold = self.similarity_threshold
        price_vectors = self.construct_price_vectors(year, category_name)
        if not price_vectors:
            return set(), {}
        relation = set()
        similarity_matrix = {}
        for city_i, city_j in combinations(self.cities, 2):
            vec_i = price_vectors.get(city_i, None)
            vec_j = price_vectors.get(city_j, None)
            if vec_i is None or vec_j is None:
                continue
            similarity = self.compute_cosine_similarity(vec_i, vec_j)
            similarity_matrix[(city_i, city_j)] = similarity
            similarity_matrix[(city_j, city_i)] = similarity
            if similarity >= threshold:
                relation.add((city_i, city_j))
                relation.add((city_j, city_i))
        for city in self.cities:
            relation.add((city, city))
            similarity_matrix[(city, city)] = 1.0
        return relation, similarity_matrix

    def construct_graph(self, year, category_name, threshold=None):
        relation, similarity_matrix = self.construct_similarity_relation(year, category_name, threshold)
        G = nx.Graph()
        G.add_nodes_from(self.cities)
        for (city_i, city_j) in relation:
            if city_i != city_j:
                weight = similarity_matrix.get((city_i, city_j), 0.0)
                if weight > 0:
                    G.add_edge(city_i, city_j, weight=weight)
        G.graph['similarity_matrix'] = similarity_matrix
        G.graph['year'] = year
        G.graph['category'] = category_name
        G.graph['threshold'] = threshold if threshold else self.similarity_threshold
        return G

    def build_all_networks(self, threshold=None):
        if self.data is None:
            return
        built_count = 0
        for year in self.years:
            for category_name in self.category_importance_weights.keys():
                key = (year, category_name)
                G = self.construct_graph(year, category_name, threshold)
                if G.number_of_edges() > 0:
                    self.networks[key] = G
                    built_count += 1

    def verify_temporal_relation(self, category_name):
        category_graphs = {}
        for year in self.years:
            key = (year, category_name)
            if key in self.networks:
                category_graphs[year] = self.networks[key]
        if len(category_graphs) == 0:
            return None, None, None
        years_list = sorted(category_graphs.keys())
        n = len(years_list)
        T = np.zeros((n, n), dtype=int)
        for i, year_i in enumerate(years_list):
            for j, year_j in enumerate(years_list):
                G_i = category_graphs[year_i]
                G_j = category_graphs[year_j]
                edges_i = set(tuple(sorted([u, v])) for u, v in G_i.edges())
                edges_j = set(tuple(sorted([u, v])) for u, v in G_j.edges())
                if edges_i.issubset(edges_j):
                    T[i][j] = 1
        results = {
            'reflexive': all(T[i][i] == 1 for i in range(n)),
            'antisymmetric': all(i == j or T[i][j] == 0 or T[j][i] == 0 for i in range(n) for j in range(n)),
            'transitive': all(T[i][k] == 1 for i in range(n) for j in range(n) for k in range(n) if T[i][j] == 1 and T[j][k] == 1)
        }
        results['is_partial_order'] = (results['reflexive'] and results['antisymmetric'] and results['transitive'])
        return results, T, years_list

    def create_hasse_diagram(self, category_name, save_path=None, figsize=(10, 8)):
        results, T, years_list = self.verify_temporal_relation(category_name)
        if results is None or not years_list:
            return

        years_str = ", ".join(map(str, sorted(years_list)))

        props_text = (
            f"Partial Order Properties:\n"
            f"• Reflexive: {results['reflexive']} ({'✓' if results['reflexive'] else '✗'})\n"
            f"• Antisymmetric: {results['antisymmetric']} ({'✓' if results['antisymmetric'] else '✗'})\n"
            f"• Transitive: {results['transitive']} ({'✓' if results['transitive'] else '✗'})\n"
            f"→ Is Partial Order: {results['is_partial_order']} ({'✓' if results['is_partial_order'] else '✗'})\n"
        )

        plt.figure(figsize=figsize)

        if results['is_partial_order']:
            # ONLY DRAW HASSE DIAGRAM IF IT IS A PARTIAL ORDER
            H = nx.DiGraph()
            for year in years_list:
                H.add_node(year)

            n = len(years_list)
            for i in range(n):
                for j in range(n):
                    if i != j and T[i][j] == 1:
                        is_direct = True
                        for k in range(n):
                            if k != i and k != j and T[i][k] == 1 and T[k][j] == 1:
                                is_direct = False
                                break
                        if is_direct:
                            H.add_edge(years_list[i], years_list[j])

            pos = nx.spring_layout(H, k=0.1, iterations=150, seed=42)

            levels = {node: 0 for node in H.nodes()}
            for node in H.nodes():
                for other in H.nodes():
                    if other != node and nx.has_path(H, other, node):
                        levels[node] = max(levels[node], nx.shortest_path_length(H, other, node))

            max_level = max(levels.values(), default=0)
            if max_level > 0:
                for node, coords in pos.items():
                    level = levels[node]
                    coords[1] = -2 * level / max_level

            node_sizes = [2000 for _ in H.nodes()]

            nx.draw_networkx_nodes(H, pos, node_color='lightblue', node_size=node_sizes,
                                    alpha=0.9, edgecolors='black', linewidths=2)
            nx.draw_networkx_labels(H, pos, font_size=10, font_weight='bold')

            if H.number_of_edges() > 0:
                nx.draw_networkx_edges(H, pos, edge_color='gray', arrows=True,
                                        arrowsize=25, width=2, alpha=0.7, connectionstyle="arc3,rad=0.1")

            title = f'Hasse Diagram: Temporal Relation (T) for {category_name}\nYears: {years_str}'

        else:
            # NOT A PARTIAL ORDER → SHOW MESSAGE ONLY, NO DIAGRAM AT ALL
            plt.text(0.5, 0.5, "NOT A\nPARTIAL ORDER\n\nNO HASSE\nDIAGRAM", 
                     ha='center', va='center', fontsize=28, color='darkred', fontweight='bold')
            title = f'No Hasse Diagram ─ Not a Partial Order: {category_name}\nYears: {years_str}'

        # ALWAYS SHOW THE PROPERTIES BOX
        plt.figtext(0.02, 0.02, props_text, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                            edgecolor="black", alpha=0.8))

        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
        
    def create_hasse_diagrams_for_all_categories(self, save_dir='results'):
        if len(self.years) < 2:
            return
        categories_with_data = []
        for category_name in self.category_importance_weights.keys():
            category_years = [y for y in self.years if (y, category_name) in self.networks]
            if len(category_years) >= 2:
                categories_with_data.append((category_name, category_years))
        if not categories_with_data:
            return

        for idx, (category_name, years) in enumerate(categories_with_data, 1):
            save_path = None
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                safe_name = category_name.replace(' & ', '').replace(' ', '').replace(',', '').replace('/', '_')
                save_path = os.path.join(save_dir, f'hasse_{safe_name}.png')
          
            self.create_hasse_diagram(category_name, save_path=save_path)
            
    def aggregate_across_categories(self, year, method='equal', custom_weights=None):
        total_scores = {city: 0.0 for city in self.cities}
        for category in self.category_importance_weights.keys():
            category_scores = None
            if method == 'equal':
                category_scores = self.aggregate_centrality_equal(year, category)
            elif method == 'correlation':
                category_scores, _ = self.aggregate_centrality_correlation(year, category)
            elif method == 'entropy':
                category_scores, _ = self.aggregate_centrality_entropy(year, category)
            elif method == 'custom':
                wD, wC, wB, wE = custom_weights if custom_weights else (0.25, 0.25, 0.25, 0.25)
                category_scores = self.aggregate_centrality_custom(year, category, wD, wC, wB, wE)
            else:
                raise ValueError("method must be 'equal', 'correlation', 'entropy' or 'custom'")
            if not category_scores:
                continue
            for city, score in category_scores.items():
                if city in total_scores:
                    total_scores[city] += score
                else:
                    total_scores[city] = score
        return total_scores
    def compute_centrality_measures(self, year, category_name):
        key = (year, category_name)
        G = self.networks.get(key)
        if G is None:
            return {}
        centralities = {}
        degree_cent = nx.degree_centrality(G)
        closeness_cent = {}
        if nx.is_connected(G):
            closeness_cent = nx.closeness_centrality(G)
        else:
            for component in nx.connected_components(G):
                subG = G.subgraph(component)
                comp_closeness = nx.closeness_centrality(subG)
                closeness_cent.update(comp_closeness)
            for city in self.cities:
                if city not in closeness_cent:
                    closeness_cent[city] = 0.0
        betweenness_cent = nx.betweenness_centrality(G)
        eigenvector_cent = {}
        for component in nx.connected_components(G):
            subG = G.subgraph(component)
            if len(subG) > 1:
                comp_eigen = nx.eigenvector_centrality_numpy(subG)
                eigenvector_cent.update(comp_eigen)
        for city in self.cities:
            if city not in eigenvector_cent:
                eigenvector_cent[city] = 0.0
        for city in self.cities:
            centralities[city] = {
                'degree': degree_cent.get(city, 0.0),
                'closeness': closeness_cent.get(city, 0.0),
                'betweenness': betweenness_cent.get(city, 0.0),
                'eigenvector': eigenvector_cent.get(city, 0.0)
            }
        self.centrality_results[key] = centralities
        return centralities

    def compute_all_centralities(self):
        for (year, category_name) in list(self.networks.keys()):
            self.compute_centrality_measures(year, category_name)

    def aggregate_centrality_custom(self, year, category_name, wD, wC, wB, wE):
        """Custom weighting WITH category importance weight ac applied"""
        key = (year, category_name)
        centralities = self.centrality_results.get(key, {})
        if not centralities:
            return {}
        total = wD + wC + wB + wE
        if total > 0:
            wD, wC, wB, wE = wD/total, wC/total, wB/total, wE/total
        else:
            wD = wC = wB = wE = 0.25
        scores = {}
        alpha_c = self.category_importance_weights[category_name]
        for city in self.cities:
            cent = centralities.get(city, {'degree': 0.0, 'closeness': 0.0, 'betweenness': 0.0, 'eigenvector': 0.0})
            score = alpha_c * (wD * cent['degree'] + wC * cent['closeness'] +
                             wB * cent['betweenness'] + wE * cent['eigenvector'])
            scores[city] = score
        return scores

    def aggregate_centrality_equal(self, year, category_name):
        key = (year, category_name)
        centralities = self.centrality_results.get(key, {})
        
        if not centralities:
            return {}
        
        scores = {}
        for city in self.cities:
            cent = centralities.get(city, {'degree': 0.0, 'closeness': 0.0, 'betweenness': 0.0, 'eigenvector': 0.0})
            alpha_c = self.category_importance_weights[category_name]
            score = alpha_c * (0.25 * cent['degree'] + 0.25 * cent['closeness'] +
                             0.25 * cent['betweenness'] + 0.25 * cent['eigenvector'])
            scores[city] = score
        return scores

    def aggregate_centrality_correlation(self, year, category_name):
        key = (year, category_name)
        centralities = self.centrality_results.get(key, {})
        
        if not centralities:
            return {}, {}
        
        measures = ['degree', 'closeness', 'betweenness', 'eigenvector']
        n_cities = len(self.cities)
        cent_matrix = np.zeros((n_cities, len(measures)))
        for i, city in enumerate(self.cities):
            for j, measure in enumerate(measures):
                cent_matrix[i, j] = centralities.get(city, {}).get(measure, 0.0)
        corr_matrix = np.corrcoef(cent_matrix.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        weights = {}
        for k, measure in enumerate(measures):
            corr_sum = sum(abs(corr_matrix[k, j]) for j in range(len(measures)) if j != k)
            weights[measure] = 1.0 / (1.0 + corr_sum)
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        else:
            weights = {k: 0.25 for k in measures}
        alpha_c = self.category_importance_weights[category_name]
        scores = {}
        for city in self.cities:
            cent = centralities.get(city, {'degree': 0.0, 'closeness': 0.0, 'betweenness': 0.0, 'eigenvector': 0.0})
            score = alpha_c * sum(weights[m] * cent[m] for m in measures)
            scores[city] = score
        return scores, weights

    def aggregate_centrality_entropy(self, year, category_name):
        key = (year, category_name)
        centralities = self.centrality_results.get(key, {})
        if not centralities:
            return {}, {}
        measures = ['degree', 'closeness', 'betweenness', 'eigenvector']
        entropies = {}
        for measure in measures:
            values = [centralities.get(city, {}).get(measure, 0.0) for city in self.cities]
            total = sum(values)
            if total == 0:
                entropies[measure] = 0.0
                continue
            probs = [v / total for v in values if v > 0]
            if not probs:
                entropies[measure] = 0.0
                continue
            epsilon = 1e-10
            H = -sum(p * np.log(p + epsilon) for p in probs)
            entropies[measure] = H
        total_entropy = sum(entropies.values())
        if total_entropy > 0:
            weights = {k: v / total_entropy for k, v in entropies.items()}
        else:
            weights = {k: 0.25 for k in measures}
        alpha_c = self.category_importance_weights[category_name]
        scores = {}
        for city in self.cities:
            cent = centralities.get(city, {'degree': 0.0, 'closeness': 0.0, 'betweenness': 0.0, 'eigenvector': 0.0})
            score = alpha_c * sum(weights[measure] * cent[measure] for measure in measures)
            scores[city] = score
        return scores, weights

    def interactive_weighting_visualization(self, year, category_name):
        key = (year, category_name)
        
        if key not in self.centrality_results:
            return
        
        fig, (ax_bars, ax_text) = plt.subplots(1, 2, figsize=(16, 8))
        plt.subplots_adjust(left=0.1, bottom=0.35, right=0.95, top=0.92)
        
        init_wD, init_wC, init_wB, init_wE = 0.25, 0.25, 0.25, 0.25

        scores = self.aggregate_centrality_custom(year, category_name, init_wD, init_wC, init_wB, init_wE)
        sorted_cities = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_10 = sorted_cities[:10]
      
        cities_list = [c[0] for c in top_10]
        scores_list = [c[1] for c in top_10]
        bars = ax_bars.barh(cities_list, scores_list, color='steelblue', alpha=0.8)
        ax_bars.set_xlabel('Aggregated Influence Score', fontsize=12, fontweight='bold')
        ax_bars.set_ylabel('Cities', fontsize=12, fontweight='bold')
        ax_bars.set_title(f'Top 10 Cities by Influence\n{category_name} ({year})',
                         fontsize=14, fontweight='bold', pad=15)
        ax_bars.invert_yaxis()
        ax_bars.grid(axis='x', alpha=0.3)
        ax_text.axis('off')
        alpha_c = self.category_importance_weights[category_name]
      
        text_content = (
            f"WEIGHTED CENTRALITY AGGREGATION\n"
            f"{'='*50}\n\n"
            f"Category: {category_name}\n"
            f"Category Weight (ac): {alpha_c:.2f}\n\n"
            f"Current Weights:\n"
            f" • Degree (wD): {init_wD:.3f}\n"
            f" • Closeness (wC): {init_wC:.3f}\n"
            f" • Betweenness (wB): {init_wB:.3f}\n"
            f" • Eigenvector (wE): {init_wE:.3f}\n\n"
            f"Formula:\n"
            f"S_i,y,c = ac × (wD·D + wC·C + wB·B + wE·E)\n\n"
            f"where Σ(wD + wC + wB + wE) = 1\n\n"
            f"Adjust sliders below to explore\n"
            f"how different weightings affect\n"
            f"city rankings and influence scores."
        )
      
        text_obj = ax_text.text(0.05, 0.95, text_content, transform=ax_text.transAxes,
                               fontsize=11, verticalalignment='top', family='monospace',
                               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        ax_wD = plt.axes([0.15, 0.25, 0.7, 0.02])
        ax_wC = plt.axes([0.15, 0.20, 0.7, 0.02])
        ax_wB = plt.axes([0.15, 0.15, 0.7, 0.02])
        ax_wE = plt.axes([0.15, 0.10, 0.7, 0.02])
        slider_wD = Slider(ax_wD, 'Degree (wD)', 0.0, 1.0, valinit=init_wD, color='steelblue')
        slider_wC = Slider(ax_wC, 'Closeness (wC)', 0.0, 1.0, valinit=init_wC, color='seagreen')
        slider_wB = Slider(ax_wB, 'Betweenness (wB)', 0.0, 1.0, valinit=init_wB, color='darkorange')
        slider_wE = Slider(ax_wE, 'Eigenvector (wE)', 0.0, 1.0, valinit=init_wE, color='purple')
        ax_reset = plt.axes([0.4, 0.03, 0.1, 0.04])
        btn_reset = Button(ax_reset, 'Reset', color='lightgray', hovercolor='gray')
        def update(val):
            wD = slider_wD.val
            wC = slider_wC.val
            wB = slider_wB.val
            wE = slider_wE.val
            scores = self.aggregate_centrality_custom(year, category_name, wD, wC, wB, wE)
            sorted_cities = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_10 = sorted_cities[:10]
          
            cities_list = [c[0] for c in top_10]
            scores_list = [c[1] for c in top_10]
            ax_bars.clear()
            ax_bars.barh(cities_list, scores_list, color='steelblue', alpha=0.8)
            ax_bars.set_xlabel('Aggregated Influence Score', fontsize=12, fontweight='bold')
            ax_bars.set_ylabel('Cities', fontsize=12, fontweight='bold')
            ax_bars.set_title(f'Top 10 Cities by Influence\n{category_name} ({year})',
                             fontsize=14, fontweight='bold', pad=15)
            ax_bars.invert_yaxis()
            ax_bars.grid(axis='x', alpha=0.3)
            total = wD + wC + wB + wE
            wD_norm = wD/total if total > 0 else 0.25
            wC_norm = wC/total if total > 0 else 0.25
            wB_norm = wB/total if total > 0 else 0.25
            wE_norm = wE/total if total > 0 else 0.25
            text_content = (
                f"WEIGHTED CENTRALITY AGGREGATION\n"
                f"{'='*50}\n\n"
                f"Category: {category_name}\n"
                f"Category Weight (ac): {alpha_c:.2f}\n\n"
                f"Current Weights (Normalized):\n"
                f" • Degree (wD): {wD_norm:.3f}\n"
                f" • Closeness (wC): {wC_norm:.3f}\n"
                f" • Betweenness (wB): {wB_norm:.3f}\n"
                f" • Eigenvector (wE): {wE_norm:.3f}\n"
                f" • Sum: {wD_norm+wC_norm+wB_norm+wE_norm:.3f}\n\n"
                f"Formula:\n"
                f"S_i,y,c = ac × (wD·D + wC·C + wB·B + wE·E)\n\n"
                f"where Σ(wD + wC + wB + wE) = 1\n\n"
                f"Adjust sliders below to explore\n"
                f"how different weightings affect\n"
                f"city rankings and influence scores."
            )
          
            text_obj.set_text(text_content)
            fig.canvas.draw_idle()
        def reset(event):
            slider_wD.reset()
            slider_wC.reset()
            slider_wB.reset()
            slider_wE.reset()
        slider_wD.on_changed(update)
        slider_wC.on_changed(update)
        slider_wB.on_changed(update)
        slider_wE.on_changed(update)
        btn_reset.on_clicked(reset)
        plt.suptitle(' INTERACTIVE WEIGHTING SCHEME EXPLORER ',
                    fontsize=16, fontweight='bold', y=0.98)
        plt.show()

    def visualize_network_enhanced(self, year, category_name, save_path=None, show_properties=True, compact=True):
        key = (year, category_name)
        G = self.networks.get(key)
        if G is None:
            return
        density = nx.density(G)
        components = nx.number_connected_components(G)
        avg_clustering = nx.average_clustering(G) if G.number_of_nodes() > 1 else 0
        threshold = G.graph.get('threshold', self.similarity_threshold)
      
        relation_props = self._check_relation_properties(G)
      
        fig, ax = plt.subplots(figsize=(10, 7) if compact else (14, 10))
      
        k_value = 0.12 if compact else 0.2
        pos = nx.spring_layout(G, k=k_value, iterations=100, seed=42, dim=2)
      
        node_sizes = [G.degree(node) * 80 + 200 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue',
                            alpha=0.85, edgecolors='black', linewidths=1.2)
        nx.draw_networkx_labels(G, pos, font_size=7, font_weight='bold')
      
        if G.number_of_edges() > 0:
            weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, edge_color='gray')
      
        if show_properties and relation_props:
            props_text = (
                f"Similarity Relation Properties:\n"
                f"• Reflexive: {relation_props['reflexive']} {'' if relation_props['reflexive'] else '✗'}\n"
                f"• Symmetric: {relation_props['symmetric']} {'' if relation_props['symmetric'] else '✗'}\n"
                f"• Transitive: {relation_props['transitive']} {'' if relation_props['transitive'] else '✗'}\n"
                f"→ Is Equivalence: {relation_props['is_equivalence']} {'' if relation_props['is_equivalence'] else '✗'}\n\n"
                f"Network Statistics:\n"
                f"• Cities: {G.number_of_nodes()}\n"
                f"• Connections: {G.number_of_edges()}\n"
                f"• Density: {density:.3f}\n"
                f"• Threshold (τ): {threshold:.2f}"
            )
            plt.figtext(0.02, 0.02, props_text, fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                                edgecolor="black", alpha=0.85))
        stats_line = f"Cities: {G.number_of_nodes()} | Connections: {G.number_of_edges()} | Density: {density:.3f}"
        plt.title(f'Price Similarity Network: {category_name} ({year})\n{stats_line}',
                fontsize=12, fontweight='bold', pad=15)
        plt.axis('off')
        plt.tight_layout()
        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
      
        plt.show()

    def _check_relation_properties(self, G):
        if not G or G.number_of_nodes() == 0:
            return None
      
        nodes = list(G.nodes())
        n = len(nodes)
        adj_matrix = np.zeros((n, n))
        node_index = {node: i for i, node in enumerate(nodes)}
      
        for u, v in G.edges():
            i, j = node_index[u], node_index[v]
            adj_matrix[i][j] = 1
            adj_matrix[j][i] = 1
      
        np.fill_diagonal(adj_matrix, 1)
      
        reflexive = np.all(np.diag(adj_matrix) == 1)
        symmetric = np.array_equal(adj_matrix, adj_matrix.T)
      
        transitive = True
       
        for i in range(n):
            for j in range(n):
                if adj_matrix[i][j] == 1:
                    for k in range(n):
                        if adj_matrix[j][k] == 1 and adj_matrix[i][k] == 0:
                            transitive = False
                            break
                    if not transitive:
                        break
            if not transitive:
                break
      
        return {
            'reflexive': reflexive,
            'symmetric': symmetric,
            'transitive': transitive,
            'is_equivalence': reflexive and symmetric and transitive
        }

    def visualize_all_category_networks_for_year(self, year, save_dir='results'):

      
        categories_with_data = []
        
        for category_name in self.category_importance_weights.keys():
            key = (year, category_name)
            if key in self.networks and self.networks[key].number_of_nodes() > 0:
                categories_with_data.append(category_name)
                
        if not categories_with_data:
            return

      
        for idx, category_name in enumerate(categories_with_data, 1):
            save_path = None
            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                safe_name = category_name.replace(' & ', '').replace(' ', '').replace(',', '').replace('/', '_')
                save_path = os.path.join(save_dir, f'network_{safe_name}_{year}.png')
          
            self.visualize_network_enhanced(year, category_name, save_path=save_path,
                                        show_properties=True, compact=True)

       
    def compare_weighting_schemes(self, year, category_name):
        equal_scores = self.aggregate_centrality_equal(year, category_name)
        corr_scores, corr_weights = self.aggregate_centrality_correlation(year, category_name)
        entropy_scores, entropy_weights = self.aggregate_centrality_entropy(year, category_name)
        comparison = pd.DataFrame({
            'City': self.cities,
            'Equal_Weight': [equal_scores.get(c, 0.0) for c in self.cities],
            'Correlation_Based': [corr_scores.get(c, 0.0) for c in self.cities],
            'Entropy_Based': [entropy_scores.get(c, 0.0) for c in self.cities]
        })
        comparison = comparison.sort_values('Equal_Weight', ascending=False)
        return comparison, corr_weights, entropy_weights
    
    def generate_comprehensive_report(self, year, save_dir='results'):
        os.makedirs(save_dir, exist_ok=True)
    
        data = []

        # Always compute overall scores — even if some categories are missing
        overall_scores = self.aggregate_across_categories(year, method='equal')
        
        # If no scores at all (very rare), create empty placeholder
        if not overall_scores or all(score == 0 for score in overall_scores.values()):
            print(f"No influence data for year {year} — creating placeholder row")
            data.append({
                'Year': year,
                'Category': 'No Data',
                'Rank': '-',
                'City': '-',
                'Influence_Score': 0.0,
                'Overall_Rank': '-',
                'Overall_Score': 0.0
            })
        else:
            # Create ranking from available scores
            sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
            city_overall_rank = {}
            rank = 1
            for city, score in sorted_overall:
                if score > 0:
                    city_overall_rank[city] = rank
                    rank += 1

            city_overall_score = {city: round(score, 5) for city, score in overall_scores.items()}

            # Per-category top cities (only if category has a network)
            has_any_category = False
            for category in sorted(self.category_importance_weights.keys()):
                cat_scores = self.aggregate_centrality_equal(year, category)
                if not cat_scores or all(s == 0 for s in cat_scores.values()):
                    continue  # skip empty categories
                has_any_category = True
                sorted_cat = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                for rank, (city, cat_score) in enumerate(sorted_cat, 1):
                    if cat_score > 0:
                        data.append({
                            'Year': year,
                            'Category': category,
                            'Rank': rank,
                            'City': city,
                            'Influence_Score': round(cat_score, 5),
                            'Overall_Rank': city_overall_rank.get(city, '-'),
                            'Overall_Score': city_overall_score.get(city, 0.0)
                        })

            # Always add overall top 10 (even if no per-category data)
            if sorted_overall:
                if data:  # add separator only if per-category exists
                    data.append({k: '' for k in data[0].keys()})
                    data[-1]['Category'] = '                               '

                for rank, (city, score) in enumerate(sorted_overall[:10], 1):
                    data.append({
                        'Year': year,
                        'Category': 'Overall Influence',
                        'Rank': rank,
                        'City': city,
                        'Influence_Score': round(score, 5),
                        'Overall_Rank': rank,
                        'Overall_Score': round(score, 5)
                    })

        # Always save the file, even if minimal
        if data:
            df = pd.DataFrame(data)
            csv_path = os.path.join(save_dir, f'city_influence_report_{year}.csv')
            df.to_csv(csv_path, index=False)
            print(f"Report generated for {year}: {csv_path}")
        else:
            print(f"No data at all for {year} — skipping CSV")


    def plot_similarity_heatmap(self, year, category_name, save_path=None, figsize=(10, 8)):
        """Plot city-by-city similarity heatmap"""
        key = (year, category_name)
        G = self.networks.get(key)
        if not G:
            print(f"No network for {category_name} in {year}")
            return

        cities = sorted(self.cities)
        n = len(cities)
        sim_matrix = np.zeros((n, n))
        similarity_dict = G.graph['similarity_matrix']

        city_to_idx = {city: i for i, city in enumerate(cities)}
        for i, city1 in enumerate(cities):
            for j, city2 in enumerate(cities):
                sim_matrix[i, j] = similarity_dict.get((city1, city2), 0.0)

        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
        sns.heatmap(sim_matrix, mask=mask, annot=False, cmap="viridis",
                    xticklabels=cities, yticklabels=cities,
                    square=True, cbar_kws={"shrink": 0.8},
                    linewidths=0.5, linecolor='gray')

        plt.title(f'Price Similarity Heatmap\n{category_name} — {year}', fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('City', fontsize=12)
        plt.ylabel('City', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_centrality_comparison_heatmap(self, year, save_dir="results"):
        """Heatmap comparing 4 centrality measures across cities for a given year"""
        os.makedirs(save_dir, exist_ok=True)

        data = []
        for category in self.category_importance_weights.keys():
            scores = self.aggregate_centrality_equal(year, category)
            if not scores:
                continue
            for city in self.cities:
                cent = self.centrality_results.get((year, category), {}).get(city, {})
                data.append({
                    'Category': category[:20],
                    'City': city,
                    'Degree': cent.get('degree', 0),
                    'Closeness': cent.get('closeness', 0),
                    'Betweenness': cent.get('betweenness', 0),
                    'Eigenvector': cent.get('eigenvector', 0),
                    'Influence': scores.get(city, 0)
                })

        if not data:
            return
        df = pd.DataFrame(data)
        pivot = df.pivot_table(index='City', columns='Category', values='Influence', aggfunc='mean', fill_value=0)

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlOrRd", linewidths=0.5, cbar_kws={"label": "Influence Score"})
        plt.title(f'City Influence Heatmap Across Categories — {year}', fontsize=16, fontweight='bold')
        plt.ylabel('City')
        plt.xlabel('Product Category')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        path = os.path.join(save_dir, f"influence_heatmap_{year}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.show()

    def plot_weighting_scheme_comparison(self, year, top_n=10, save_path=None):
        """Bar plot comparing different aggregation methods"""
        methods = ['Equal', 'Correlation_Based', 'Entropy_Based']
        results = {}

        for category in self.category_importance_weights.keys():
            equal = self.aggregate_centrality_equal(year, category)
            corr, _ = self.aggregate_centrality_correlation(year, category)
            ent, _ = self.aggregate_centrality_entropy(year, category)

            for city in self.cities:
                results.setdefault(city, {'Equal': 0, 'Correlation_Based': 0, 'Entropy_Based': 0})
                results[city]['Equal'] += equal.get(city, 0)
                results[city]['Correlation_Based'] += corr.get(city, 0)
                results[city]['Entropy_Based'] += ent.get(city, 0)

        df = pd.DataFrame(results).T
        df = df.sort_values('Equal', ascending=False).head(top_n)

        df.plot(kind='bar', figsize=(14, 8), width=0.8, cmap="Set2")
        plt.title(f'City Influence: Weighting Scheme Comparison ({year})', fontsize=16, fontweight='bold')
        plt.xlabel('City')
        plt.ylabel('Total Influence Score')
        plt.legend(title="Weighting Method")
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_rank_evolution_over_time(self, city=None, save_path=None):
        """Line plot showing how a city's overall rank changes over years"""
        years = sorted(self.years)
        if len(years) < 2:
            return

        city_ranks = {}
        for year in years:
            scores = self.aggregate_across_categories(year, method='equal')
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (c, _) in enumerate(ranked, 1):
                if c not in city_ranks:
                    city_ranks[c] = []
                city_ranks[c].append((year, rank))

        plt.figure(figsize=(12, 7))
        for c, history in city_ranks.items():
            if city is None or c == city:
                ys = [rank for _, rank in history]
                xs = [y for y, _ in history]
                plt.plot(xs, ys, marker='o', label=c, linewidth=2.5)

        plt.gca().invert_yaxis()
        plt.title('City Influence Rank Evolution Over Time', fontsize=16, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Rank (1 = Most Influential)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_category_dashboard(self, year, save_dir="results"):
        """Grid of all category networks + stats in one figure"""
        valid_categories = [cat for cat in self.category_importance_weights if (year, cat) in self.networks]
        n = len(valid_categories)
        if n == 0:
            return

        cols = 3
        rows = (n + cols - 1) // cols
        fig = plt.figure(figsize=(cols*6, rows*5))

        for idx, cat in enumerate(valid_categories):
            ax = fig.add_subplot(rows, cols, idx + 1)
            G = self.networks[(year, cat)]
            pos = nx.spring_layout(G, seed=42, k=0.2)
            sizes = [G.degree(n) * 100 + 200 for n in G.nodes()]
            nx.draw(G, pos, ax=ax, node_size=sizes, node_color='skyblue',
                    edge_color='gray', alpha=0.7, with_labels=True, font_size=8)
            ax.set_title(f"{cat}\nEdges: {G.number_of_edges()}", fontsize=10)

        plt.suptitle(f'Price Similarity Networks by Category — {year}', fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()
        path = os.path.join(save_dir, f"category_dashboard_{year}.png")
        plt.savefig(path, bbox_inches='tight')
        plt.show()

    def plot_temporal_influence_heatmap(self, save_path=None):
        """Heatmap: Cities (rows) × Years (columns) → Influence Score"""
        data = []
        for year in sorted(self.years):
            scores = self.aggregate_across_categories(year, method='equal')
            for city in self.cities:
                data.append({'Year': year, 'City': city, 'Influence': scores.get(city, 0)})

        df = pd.DataFrame(data)
        pivot = df.pivot(index='City', columns='Year', values='Influence')

        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdYlBu_r", linewidths=0.5,
                    cbar_kws={"label": "Influence Score"})
        plt.title('Temporal Evolution of City Influence (All Categories)', fontsize=16, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('City')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def plot_centrality_correlation_heatmap(self, year, category_name, save_path=None):
        """Correlation between centrality measures"""
        key = (year, category_name)
        if key not in self.centrality_results:
            return

        measures = ['degree', 'closeness', 'betweenness', 'eigenvector']
        matrix = np.zeros((4, 4))
        names = ['Degree', 'Closeness', 'Betweenness', 'Eigenvector']

        vals = {m: [] for m in measures}
        for city in self.cities:
            c = self.centrality_results[key].get(city, {})
            for m in measures:
                vals[m].append(c.get(m, 0))

        for i, m1 in enumerate(measures):
            for j, m2 in enumerate(measures):
                corr = np.corrcoef(vals[m1], vals[m2])[0,1]
                matrix[i,j] = corr if not np.isnan(corr) else 0

        plt.figure(figsize=(7, 6))
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    xticklabels=names, yticklabels=names, square=True,
                    cbar_kws={"shrink": 0.8})
        plt.title(f'Centrality Measure Correlations\n{category_name} — {year}')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()




def compare_all_categories(analyzer, year):
    comparison_data = []
    for category in sorted(analyzer.category_importance_weights.keys()):
        key = (year, category)
        if key in analyzer.networks:
            G = analyzer.networks[key]
            products = analyzer.get_products_by_category(category)
            comparison_data.append({
                'Category': category,
                'ac_weight': analyzer.category_importance_weights[category],
                'Products': len(products),
                'Edges': G.number_of_edges(),
                'Density': f"{nx.density(G):.3f}",
                'Clustering': f"{nx.average_clustering(G):.3f}",
                'Components': nx.number_connected_components(G)
            })
            
    if comparison_data:
        df = pd.DataFrame(comparison_data)
        df['Edges_num'] = df['Edges'].astype(int)
        most_connected = df.loc[df['Edges_num'].idxmax()]
        least_connected = df.loc[df['Edges_num'].idxmin()]



def main():
    analyzer = CPINetworkAnalyzer()
    all_csvs = glob.glob('*.csv')
    if not all_csvs:
        print("No CSV files found!")
        return

    analyzer.load_data(all_csvs)
    if analyzer.data is None or analyzer.data.empty:
        print("No valid data loaded.")
        return

    print(f"Loaded {len(analyzer.data)} price records across {len(analyzer.cities)} cities")

    analyzer.build_all_networks(threshold=0.7)
    if analyzer.networks:
        analyzer.compute_all_centralities()

    os.makedirs("results", exist_ok=True)
    os.makedirs("results/heatmaps", exist_ok=True)
    os.makedirs("results/comparisons", exist_ok=True)

    # Generate all visualizations
    for year in analyzer.years:
        print(f"\nGenerating visuals for {year}...")

        # 1. Heatmaps
        for cat in analyzer.category_importance_weights:
            if (year, cat) in analyzer.networks:
                analyzer.plot_similarity_heatmap(year, cat,
                    save_path=f"results/heatmaps/similarity_{cat.replace(' ', '_')}_{year}.png")

        # 2. Dashboard
        analyzer.plot_category_dashboard(year)

        # 3. Influence Heatmap
        analyzer.plot_centrality_comparison_heatmap(year)

        # 4. Weighting Comparison
        analyzer.plot_weighting_scheme_comparison(year,
            save_path=f"results/comparisons/weighting_comparison_{year}.png")

        # 5. Reports
        analyzer.generate_comprehensive_report(year)
        
        for cat in analyzer.category_importance_weights.keys():
            key = (year, cat) 
            if key in analyzer.centrality_results: 
                analyzer.interactive_weighting_visualization(year, cat)
    
    
    # 6. Temporal & Rank Evolution
    if len(analyzer.years) > 1:
        analyzer.plot_temporal_influence_heatmap(save_path="results/temporal_influence_heatmap.png")
        analyzer.plot_rank_evolution_over_time(save_path="results/rank_evolution.png")

    # 7. Hasse Diagrams
    analyzer.create_hasse_diagrams_for_all_categories(save_dir='results/hasse')

    print("\nAll visualizations completed! Check the 'results/' folder.")
  
  
if __name__ == "__main__":
    main()