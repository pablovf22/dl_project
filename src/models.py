import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_add_pool, SAGEConv, GINConv


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.linear(x)

        return x


class GCN_v2(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, dropout=0.05):
        super(GCN_v2, self).__init__()

        # Capas GCN
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        # Cabeza MLP final
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 1. Capas GCN + BatchNorm + ReLU
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # 2. Readout: pooling por suma
        x = global_add_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Cabeza MLP final
        x = self.mlp(x)  # [batch_size, 1]

        return x


class GraphSAGE(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, dropout=0.05):
        super(GraphSAGE, self).__init__()

        # Capas SAGEConv
        self.conv1 = SAGEConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        # Cabeza MLP final
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Capas SAGE + BatchNorm + ReLU
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Pooling por suma
        x = global_add_pool(x, batch)  # [batch_size, hidden_channels]

        # Cabeza MLP final
        x = self.mlp(x)  # [batch_size, 1]

        return x


class GIN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, dropout=0.05):
        super(GIN, self).__init__()

        # MLP para la primera capa (num_node_features → hidden)
        mlp1 = torch.nn.Sequential(
            torch.nn.Linear(num_node_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv1 = GINConv(mlp1)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        # MLP para la segunda capa (hidden → hidden)
        mlp2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv2 = GINConv(mlp2)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        # MLP para la tercera capa (hidden → hidden)
        mlp3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels),
        )
        self.conv3 = GINConv(mlp3)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        # Cabeza MLP final
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Capas GIN + BatchNorm + ReLU
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Pooling por suma
        x = global_add_pool(x, batch)  # [batch_size, hidden_channels]

        # Cabeza MLP final
        x = self.mlp(x)  # [batch_size, 1]

        return x


class GCN_v3(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=128, dropout=0.05):
        super(GCN_v3, self).__init__()

        # Capas GCN
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)

        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.bn3 = torch.nn.BatchNorm1d(hidden_channels)

        # Dropout interno
        self.dropout = torch.nn.Dropout(dropout)

        # Cabeza MLP final: entrada 2 * hidden (sum + mean pool concatenados)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_channels, 1),
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Primera capa (sin residual previo)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Segunda capa con skip connection
        res = x
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + res  # residual

        # Tercera capa con skip connection
        res = x
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = x + res  # residual

        # Pooling: suma y media, y los concatenamos
        x_sum = global_add_pool(x, batch)       # [batch_size, hidden]
        x_mean = global_mean_pool(x, batch)     # [batch_size, hidden]
        x = torch.cat([x_sum, x_mean], dim=-1)  # [batch_size, 2 * hidden]

        # Cabeza MLP
        x = self.mlp(x)  # [batch_size, 1]

        return x