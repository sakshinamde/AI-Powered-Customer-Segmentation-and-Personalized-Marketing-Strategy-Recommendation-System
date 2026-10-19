export interface Customer {
  customer_id: string;
  age: number;
  income: number;
  satisfaction: number;
  browsing_frequency: number;
  purchase_count: number;
  clusterId?: number;
  clusterLabel?: string;
  [key: string]: any;
}

export interface ClusterSummary {
  clusterId: number;
  name: string;
  count: number;
  percentage?: number;
  avgSatisfaction: number;
  avgIncome?: number;
  avgAge?: number;
  avgBrowsing?: number;
  avgPurchases?: number;
  avgCartCompletion?: number;
  avgReviewImportance?: number;
  topCategory?: string;
  topAbandonmentReason?: string;
  topServiceAppreciation?: string;
  topImprovementArea?: string;
}

export interface ClusteringResponse {
  algorithm: string;
  k: number;
  silhouette: number;
  totalCustomers: number;
  activeClustersCount: number;
  labels: number[];
  projectionX: number[];
  projectionY: number[];
  clusterSummaries: ClusterSummary[];
  xAxis: string;
  yAxis: string;
}

export interface XAIResponse {
  customerId: string;
  clusterId: number;
  clusterName: string;
  contributions: {
    feature: string;
    label: string;
    value: number;
    globalAverage: number;
    zScore: number;
    importance: number;
    direction: string;
  }[];
  primaryFactors: string[];
  summaryText: string;
  recommendedAction: string;
}

export interface StrategyItem {
  clusterId: number;
  title: string;
  strategy: string;
  targetAudience?: string;
  actionItems?: string[];
  updatedAt?: string;
}

export interface ModelComparisonItem {
  method: string;
  name: string;
  silhouette: number;
  status?: 'ok' | 'failed';
  error?: string;
  clustersFound?: number;
  category?: string;
  description?: string;
  isBest?: boolean;
}


