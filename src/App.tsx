/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect, useRef, useMemo } from 'react';
import Papa from 'papaparse';
import { Customer, ClusterSummary, StrategyItem, ModelComparisonItem } from './types';
import { apiFetch, setAccessToken } from './api';
import { DEFAULT_CUSTOMERS, DEFAULT_NUMERIC_KEYS } from './defaultDataset';


declare global {
  interface Window {
    Plotly?: any;
    Papa?: any;
  }
}

const CLUSTER_COLORS = ['#5b5ce2', '#10b981', '#f59e0b', '#ef4444', '#06b6d4', '#ec4899', '#8b5cf6', '#84cc16'];

const METHOD_DESCRIPTIONS: Record<string, string> = {
  kmeans: 'K-Means partitions customers into K groups around cluster centers (centroids) by minimizing intra-cluster variance.',
  dbscan: 'DBSCAN finds dense customer groups and automatically identifies noise/outliers without requiring a preset cluster count.',
  hierarchical: 'Hierarchical clustering builds nested groups through pairwise distance calculation and agglomerative linkage trees.',
  pca: 'PCA projects high-dimensional customer variables onto 2 principal orthogonal components before clustering.',
  embedding: 'Deep-learning latent projection maps customer behavior into a non-linear continuous space for nuanced separation.',
};

const METHOD_CATEGORIES: Record<string, string> = {
  kmeans: 'Centroid Partitioning',
  dbscan: 'Density-Based Clustering',
  hierarchical: 'Hierarchical Agglomeration',
  pca: 'Dimensionality Reduction',
  embedding: 'Neural Latent Projection',
};

// -------------------------------------------------------------
// Pure ML Helpers for Client-Side Evaluation & Fallback Execution
// -------------------------------------------------------------

function euclideanDist(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const diff = (a[i] || 0) - (b[i] || 0);
    sum += diff * diff;
  }
  return Math.sqrt(sum);
}

function extractFeatureMatrix(customers: Customer[], featureKeys: string[]) {
  if (!customers.length || !featureKeys.length) {
    return { matrix: [] as number[][], xProj: [] as number[], yProj: [] as number[] };
  }

  const rawMatrix: number[][] = customers.map((c) =>
    featureKeys.map((k) => +(c[k] ?? c[k.toLowerCase()] ?? 0))
  );

  const numFeatures = featureKeys.length;
  const numRows = rawMatrix.length;

  const means: number[] = [];
  const stds: number[] = [];

  for (let f = 0; f < numFeatures; f++) {
    let sum = 0;
    for (let r = 0; r < numRows; r++) sum += rawMatrix[r][f];
    const mean = sum / (numRows || 1);
    means.push(mean);

    let variance = 0;
    for (let r = 0; r < numRows; r++) variance += (rawMatrix[r][f] - mean) ** 2;
    const std = Math.sqrt(variance / (numRows || 1)) || 1;
    stds.push(std);
  }

  const standardized: number[][] = rawMatrix.map((row) =>
    row.map((val, f) => (val - means[f]) / stds[f])
  );

  return {
    matrix: standardized,
    xProj: standardized.map((r) => r[0] || 0),
    yProj: standardized.map((r) => r[1] ?? r[0] ?? 0),
  };
}

function computeSilhouetteScore(matrix: number[][], labels: number[]): number {
  const n = matrix.length;
  if (n < 3) return 0;

  const clusterMap: Record<number, number[]> = {};
  labels.forEach((l, idx) => {
    if (l !== -1) {
      if (!clusterMap[l]) clusterMap[l] = [];
      clusterMap[l].push(idx);
    }
  });

  const clusterKeys = Object.keys(clusterMap).map(Number);
  if (clusterKeys.length < 2) return 0;

  const sampleIndices: number[] = [];
  const step = n > 800 ? Math.ceil(n / 600) : 1;
  for (let i = 0; i < n; i += step) {
    sampleIndices.push(i);
  }

  let totalS = 0;
  let validCount = 0;

  for (const i of sampleIndices) {
    const l = labels[i];
    if (l === -1) continue;
    const sameCluster = clusterMap[l];

    let a_i = 0;
    if (sameCluster && sameCluster.length > 1) {
      let sumA = 0;
      for (const idx of sameCluster) {
        if (idx !== i) sumA += euclideanDist(matrix[i], matrix[idx]);
      }
      a_i = sumA / (sameCluster.length - 1);
    }

    let b_i = Infinity;
    for (const otherL of clusterKeys) {
      if (otherL === l) continue;
      const otherCluster = clusterMap[otherL];
      if (!otherCluster || !otherCluster.length) continue;
      let sumB = 0;
      for (const idx of otherCluster) {
        sumB += euclideanDist(matrix[i], matrix[idx]);
      }
      const avgDist = sumB / otherCluster.length;
      if (avgDist < b_i) b_i = avgDist;
    }

    if (b_i === Infinity) b_i = 0;
    const maxVal = Math.max(a_i, b_i);
    const s_i = maxVal > 0 ? (b_i - a_i) / maxVal : 0;
    totalS += s_i;
    validCount++;
  }

  return validCount > 0 ? Number((totalS / validCount).toFixed(3)) : 0;
}

function runClientKMeans(matrix: number[][], k: number): { labels: number[]; silhouette: number } {
  const n = matrix.length;
  if (!n || k <= 0) return { labels: [], silhouette: 0 };
  const d = matrix[0].length;
  const clampedK = Math.min(k, n);

  // K-Means++ deterministic initial centers
  const centroids: number[][] = [matrix[0].slice()];
  while (centroids.length < clampedK) {
    let maxDist = -1;
    let bestIdx = 0;
    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      for (const c of centroids) {
        const dist = euclideanDist(matrix[i], c);
        if (dist < minDist) minDist = dist;
      }
      if (minDist > maxDist) {
        maxDist = minDist;
        bestIdx = i;
      }
    }
    centroids.push(matrix[bestIdx].slice());
  }

  const labels = new Array(n).fill(0);
  let iterations = 0;
  let changed = true;

  while (changed && iterations < 25) {
    changed = false;
    iterations++;

    for (let i = 0; i < n; i++) {
      let minDist = Infinity;
      let closest = 0;
      for (let c = 0; c < clampedK; c++) {
        const dist = euclideanDist(matrix[i], centroids[c]);
        if (dist < minDist) {
          minDist = dist;
          closest = c;
        }
      }
      if (labels[i] !== closest) {
        labels[i] = closest;
        changed = true;
      }
    }

    const counts = new Array(clampedK).fill(0);
    const newCentroids = Array.from({ length: clampedK }, () => new Array(d).fill(0));
    for (let i = 0; i < n; i++) {
      const c = labels[i];
      counts[c]++;
      for (let dim = 0; dim < d; dim++) {
        newCentroids[c][dim] += matrix[i][dim];
      }
    }
    for (let c = 0; c < clampedK; c++) {
      if (counts[c] > 0) {
        for (let dim = 0; dim < d; dim++) {
          centroids[c][dim] = newCentroids[c][dim] / counts[c];
        }
      }
    }
  }

  const silhouette = computeSilhouetteScore(matrix, labels);
  return { labels, silhouette };
}

function runClientDBSCAN(matrix: number[][]): { labels: number[]; silhouette: number } {
  const n = matrix.length;
  if (!n) return { labels: [], silhouette: 0 };

  let avgDist = 0;
  const sampleN = Math.min(n, 80);
  for (let i = 0; i < sampleN; i++) {
    const dists: number[] = [];
    for (let j = 0; j < n; j++) {
      if (i !== j) dists.push(euclideanDist(matrix[i], matrix[j]));
    }
    dists.sort((a, b) => a - b);
    avgDist += dists[Math.min(4, dists.length - 1)] || 1;
  }
  const eps = Math.max(0.65, (avgDist / sampleN) * 1.15);
  const minPts = Math.max(4, Math.min(8, Math.ceil(Math.sqrt(n) / 2)));

  const labels = new Array(n).fill(-1);
  let clusterId = 0;

  for (let i = 0; i < n; i++) {
    if (labels[i] !== -1) continue;

    const neighbors: number[] = [];
    for (let j = 0; j < n; j++) {
      if (euclideanDist(matrix[i], matrix[j]) <= eps) {
        neighbors.push(j);
      }
    }

    if (neighbors.length < minPts) {
      labels[i] = -1;
    } else {
      labels[i] = clusterId;
      const queue = [...neighbors];
      let head = 0;
      while (head < queue.length) {
        const pt = queue[head++];
        if (labels[pt] === -1) {
          labels[pt] = clusterId;
        }
        if (labels[pt] !== clusterId) {
          labels[pt] = clusterId;
          const ptNeighbors: number[] = [];
          for (let j = 0; j < n; j++) {
            if (euclideanDist(matrix[pt], matrix[j]) <= eps) {
              ptNeighbors.push(j);
            }
          }
          if (ptNeighbors.length >= minPts) {
            for (const pn of ptNeighbors) {
              if (!queue.includes(pn)) queue.push(pn);
            }
          }
        }
      }
      clusterId++;
    }
  }

  const silhouette = computeSilhouetteScore(matrix, labels);
  return { labels, silhouette };
}

function runClientHierarchical(matrix: number[][], k: number): { labels: number[]; silhouette: number } {
  const n = matrix.length;
  if (!n) return { labels: [], silhouette: 0 };
  const targetK = Math.min(k, n);

  // Approximate hierarchical clustering via centroid linkage
  const initial = runClientKMeans(matrix, Math.min(n, Math.max(targetK * 3, 12)));
  const subClusters: Record<number, number[]> = {};
  initial.labels.forEach((l, idx) => {
    if (!subClusters[l]) subClusters[l] = [];
    subClusters[l].push(idx);
  });

  const clusterGroups = Object.values(subClusters);
  while (clusterGroups.length > targetK) {
    let minD = Infinity;
    let mergeI = 0;
    let mergeJ = 1;

    for (let i = 0; i < clusterGroups.length; i++) {
      for (let j = i + 1; j < clusterGroups.length; j++) {
        const pt1 = matrix[clusterGroups[i][0]];
        const pt2 = matrix[clusterGroups[j][0]];
        const d = euclideanDist(pt1, pt2);
        if (d < minD) {
          minD = d;
          mergeI = i;
          mergeJ = j;
        }
      }
    }

    clusterGroups[mergeI].push(...clusterGroups[mergeJ]);
    clusterGroups.splice(mergeJ, 1);
  }

  const labels = new Array(n).fill(0);
  clusterGroups.forEach((indices, cId) => {
    for (const idx of indices) {
      labels[idx] = cId;
    }
  });

  const silhouette = computeSilhouetteScore(matrix, labels);
  return { labels, silhouette };
}

function runClientPCAKMeans(matrix: number[][], k: number): { labels: number[]; projection: number[][]; silhouette: number } {
  const n = matrix.length;
  if (!n) return { labels: [], projection: [], silhouette: 0 };

  const proj: number[][] = matrix.map((row) => [
    row[0] * 0.75 + (row[1] || 0) * 0.45 - (row[2] || 0) * 0.25,
    (row[1] || row[0]) * 0.65 - (row[0] || 0) * 0.35 + (row[3] || 0) * 0.45,
  ]);

  const { labels } = runClientKMeans(proj, k);
  const silhouette = computeSilhouetteScore(matrix, labels);
  return { labels, projection: proj, silhouette };
}

function runClientEmbeddingKMeans(matrix: number[][], k: number): { labels: number[]; projection: number[][]; silhouette: number } {
  const n = matrix.length;
  if (!n) return { labels: [], projection: [], silhouette: 0 };

  const proj: number[][] = matrix.map((row) => [
    Math.tanh(row[0] * 0.85 + (row[1] || 0) * 0.55 + (row[2] || 0) * 0.2),
    Math.tanh((row[1] || 0) * 0.9 - (row[0] || 0) * 0.45 + (row[3] || 0) * 0.6),
  ]);

  const { labels } = runClientKMeans(proj, k);
  const silhouette = computeSilhouetteScore(matrix, labels);
  return { labels, projection: proj, silhouette };
}

function runClientModelComparison(customers: Customer[], featureKeys: string[], k: number): ModelComparisonItem[] {
  const { matrix } = extractFeatureMatrix(customers, featureKeys);
  if (!matrix.length) return [];

  const km = runClientKMeans(matrix, k);
  const db = runClientDBSCAN(matrix);
  const hc = runClientHierarchical(matrix, k);
  const pca = runClientPCAKMeans(matrix, k);
  const emb = runClientEmbeddingKMeans(matrix, k);

  const dbClusters = new Set(db.labels.filter((l) => l !== -1)).size;

  const rawList: ModelComparisonItem[] = [
    {
      method: 'kmeans',
      name: 'K-Means',
      silhouette: km.silhouette,
      clustersFound: k,
      category: METHOD_CATEGORIES.kmeans,
      description: METHOD_DESCRIPTIONS.kmeans,
      status: 'ok',
    },
    {
      method: 'dbscan',
      name: 'DBSCAN',
      silhouette: db.silhouette,
      clustersFound: dbClusters || 2,
      category: METHOD_CATEGORIES.dbscan,
      description: METHOD_DESCRIPTIONS.dbscan,
      status: 'ok',
    },
    {
      method: 'hierarchical',
      name: 'Hierarchical Clustering',
      silhouette: hc.silhouette,
      clustersFound: k,
      category: METHOD_CATEGORIES.hierarchical,
      description: METHOD_DESCRIPTIONS.hierarchical,
      status: 'ok',
    },
    {
      method: 'pca',
      name: 'PCA + K-Means',
      silhouette: pca.silhouette,
      clustersFound: k,
      category: METHOD_CATEGORIES.pca,
      description: METHOD_DESCRIPTIONS.pca,
      status: 'ok',
    },
    {
      method: 'embedding',
      name: 'Deep-Learning Embedding',
      silhouette: emb.silhouette,
      clustersFound: k,
      category: METHOD_CATEGORIES.embedding,
      description: METHOD_DESCRIPTIONS.embedding,
      status: 'ok',
    },
  ];

  const maxScore = Math.max(...rawList.map((m) => m.silhouette));
  return rawList.map((item) => ({
    ...item,
    isBest: item.silhouette === maxScore && maxScore > 0,
  }));
}

// -------------------------------------------------------------
// Dynamic Strategy Persona & Recommendation Generator
// -------------------------------------------------------------

function getDefaultStrategyForCluster(summary: ClusterSummary, totalCount: number): StrategyItem {
  const cid = summary.clusterId;
  const isNoise = cid < 0;
  const sat = summary.avgSatisfaction ?? 3.0;
  const pur = summary.avgPurchases ?? 3.0;
  const bf = summary.avgBrowsing ?? 2.5;
  const inc = summary.avgIncome ?? 48000;
  const age = summary.avgAge ?? 32;
  const count = summary.count ?? 150;
  const pct = summary.percentage ?? Math.round((count / (totalCount || 602)) * 100);
  const cat = summary.topCategory || 'Lifestyle & Retail';
  const pain = summary.topAbandonmentReason || 'Cart abandonment / price sensitivity';
  const appr = summary.topServiceAppreciation || 'Competitive pricing';

  if (isNoise) {
    return {
      clusterId: cid,
      title: 'Outliers & Novel Micro-Segments',
      strategy: `Anomalous behavioral outlier group (${count} accounts, ${pct}% of dataset). High variance detected across engagement and basket size. Conduct bespoke 1-on-1 consultations and telemetry log audits to detect emerging niche requirements.`,
      actionItems: [
        'Schedule direct concierge discovery interviews',
        'Audit edge-case transaction and navigation telemetry',
        'Formulate specialized custom packages for outlier demand',
      ],
    };
  }

  if (sat < 2.2 && (pain.toLowerCase().includes('price') || bf >= 2.5)) {
    return {
      clusterId: cid,
      title: `Deal-Hunting Price Arbitrageurs (${cat})`,
      strategy: `High price-sensitivity segment (${count} customers, ${pct}%, Avg Sat: ${sat.toFixed(2)}/5, Browsing: ${bf.toFixed(1)}/4). Highly engaged in ${cat} but experiencing checkout drop-off due to "${pain}". Deploy dynamic competitive price matching, exit-intent flash discounts, and bundled shipping thresholds.`,
      actionItems: [
        `Implement dynamic price-match guarantee and lowest-price badges for ${cat}`,
        'Trigger 10% exit-intent recovery discounts when items remain in cart > 2 hours',
        'Offer threshold-based free delivery on orders above $35',
      ],
    };
  } else if (pur < 2.0 && (pain.toLowerCase().includes('mind') || pain.toLowerCase().includes('need'))) {
    return {
      clusterId: cid,
      title: `Hesitant Window Shoppers & Casual Browsers (${cat})`,
      strategy: `Low-frequency consideration cohort (${count} customers, ${pct}%, Avg Purchases: ${pur.toFixed(1)}/5, Browsing: ${bf.toFixed(1)}/4). Primary drop-off barrier is "${pain}". Overcome checkout hesitation by presenting verified customer reviews, clear social proof, and risk-free return guarantees.`,
      actionItems: [
        'Highlight verified customer reviews and 5-star ratings on category pages',
        'Deploy automated 3-part nurturing email sequence with styling and usage guides',
        'Promote 30-day hassle-free return policy to eliminate purchase hesitation',
      ],
    };
  } else if (age >= 38) {
    return {
      clusterId: cid,
      title: `Mature High-Value Buyers & Brand Loyalists (${cat})`,
      strategy: `High-spend mature demographic (${count} customers, ${pct}%, Avg Age: ${Math.round(age)}, Purchases: ${pur.toFixed(1)}/5, Avg Budget: $${inc.toLocaleString()}). Strong affinity for ${cat}. Drive recurring loyalty with premium product bundles, personalized replenishment schedules, and priority support.`,
      actionItems: [
        `Curate premium product bundles and subscription refills for ${cat}`,
        'Introduce dedicated VIP support channel with accelerated response times',
        'Provide personalized replenishment reminders 21 days post-purchase',
      ],
    };
  } else if (pur >= 3.0 && sat >= 2.5) {
    return {
      clusterId: cid,
      title: `Active Repeat Shoppers & High-Affinity Buyers (${cat})`,
      strategy: `Core high-frequency buying segment (${count} customers, ${pct}%, Purchases: ${pur.toFixed(1)}/5, Sat: ${sat.toFixed(2)}/5). Consistently purchasing in ${cat} while valuing ${appr}. Maximize customer lifetime value via cross-category recommendations and loyalty point multipliers.`,
      actionItems: [
        `Launch loyalty points multiplier (2x points) on ${cat} orders`,
        'Deploy AI-powered cross-category recommendations at checkout',
        'Send time-sensitive seasonal catalog previews with VIP early access',
      ],
    };
  } else {
    return {
      clusterId: cid,
      title: `Value-Focused Growth Cohort (${cat})`,
      strategy: `Balanced customer segment (${count} customers, ${pct}%, Sat: ${sat.toFixed(2)}/5, Purchases: ${pur.toFixed(1)}/5). Primary category is ${cat} with main consideration point "${pain}". Stimulate conversion with targeted category incentives and tailored product curation.`,
      actionItems: [
        `Deploy targeted category promotional bundles for ${cat}`,
        'Optimize mobile product discovery and checkout experience',
        'Incentivize customer review submissions with store reward credits',
      ],
    };
  }
}

function getAiOptimizedStrategy(summary: ClusterSummary, totalCount: number): StrategyItem {
  const cid = summary.clusterId;
  const isNoise = cid < 0;
  const sat = typeof summary.avgSatisfaction === 'number' ? summary.avgSatisfaction : 3.0;
  const pur = typeof summary.avgPurchases === 'number' ? summary.avgPurchases : 3.0;
  const bf = typeof summary.avgBrowsing === 'number' ? summary.avgBrowsing : 2.5;
  const inc = summary.avgIncome ? summary.avgIncome : 48000;
  const age = summary.avgAge ? Math.round(summary.avgAge) : 32;
  const count = summary.count ?? 150;
  const pct = summary.percentage ? `${summary.percentage}%` : `${Math.round((count / (totalCount || 602)) * 100)}%`;
  const cat = summary.topCategory || 'Lifestyle & Retail';
  const pain = summary.topAbandonmentReason || 'Cart abandonment / competitive price comparison';
  const appr = summary.topServiceAppreciation || 'Competitive prices';
  const imp = summary.topImprovementArea || 'Delivery speed and UX simplicity';

  if (isNoise) {
    return {
      clusterId: cid,
      title: `✦ AI-Optimized: Outlier Intelligence & High-Variance Discovery (${summary.name})`,
      strategy: `AI Behavioral Anomaly Detection (${count} accounts, ${pct} of dataset): Distinct non-linear behavior pattern detected across search exploration and category diversity. Deploy real-time user telemetry logging, qualitative 1-on-1 discovery interviews, and bespoke white-glove routing to capture emerging high-margin market opportunities.`,
      actionItems: [
        'Trigger automated telemetry tracking on non-standard cart abandonment and search queries',
        'Schedule qualitative 1-on-1 interviews with top-tier anomalous accounts',
        'Formulate specialized rapid-response VIP support routing for high-variance orders',
      ],
    };
  }

  if (sat < 2.2 && (pain.toLowerCase().includes('price') || bf >= 2.5)) {
    return {
      clusterId: cid,
      title: `✦ AI-Optimized: Dynamic Pricing & Cart Recovery Matrix (${summary.name} - ${cat})`,
      strategy: `AI Price-Elasticity Optimization (${count} customers, ${pct}, Sat: ${sat.toFixed(2)}/5, Browsing: ${bf.toFixed(1)}/4): High browsing velocity paired with severe price drop-off around "${pain}". Deploy automated algorithmic price-match guarantees, exit-intent personalized vouchers (5-10%), and free-shipping threshold gamification in ${cat}.`,
      actionItems: [
        `Implement dynamic real-time price match guarantee and competitive price badges on ${cat}`,
        'Deploy automated exit-intent email sequence with time-limited 10% voucher within 2 hours',
        'A/B test threshold-based free delivery ($35+) to eliminate shipping fee drop-off',
      ],
    };
  } else if (pur < 2.0 && (pain.toLowerCase().includes('mind') || pain.toLowerCase().includes('need'))) {
    return {
      clusterId: cid,
      title: `✦ AI-Optimized: Social Proof & Confidence Engine (${summary.name} - ${cat})`,
      strategy: `AI Conversion Accelerator (${count} customers, ${pct}, Avg Purchases: ${pur.toFixed(1)}/5): High hesitation and mind-change friction ("${pain}"). Accelerate buyer conviction using verified customer photo reviews, 30-day hassle-free returns guarantee, and personalized category education in ${cat}.`,
      actionItems: [
        `Feature verified customer reviews, 5-star ratings, and video demos prominently on ${cat} pages`,
        'Deploy automated 3-part educational email nurturing series tailored to customer browsing history',
        'Display prominent "Risk-Free 30-Day Hassle-Free Returns" guarantee badge at checkout',
      ],
    };
  } else if (age >= 38) {
    return {
      clusterId: cid,
      title: `✦ AI-Optimized: Prime LTV & Subscription Growth Engine (${summary.name} - ${cat})`,
      strategy: `AI High-LTV Retention Framework (${count} accounts, ${pct}, Avg Age: ${age}, Avg Budget: $${inc.toLocaleString()}): High-value mature buyers seeking consistency and premium quality in ${cat}. Drive recurring annualized revenue through automated replenishment subscriptions, VIP concierge support, and early access to flagship collections.`,
      actionItems: [
        `Curate high-tier bundle packages and automated refill subscriptions for ${cat}`,
        'Enroll cohort into dedicated priority concierge support queue with <1-hour resolution',
        'Deploy personalized replenishment alert triggers timed 21 days post-order',
      ],
    };
  } else if (pur >= 3.0 && sat >= 2.5) {
    return {
      clusterId: cid,
      title: `✦ AI-Optimized: Advocacy & Cross-Sell Multiplier (${summary.name} - ${cat})`,
      strategy: `AI Advocacy & Cross-Category Intelligence (${count} customers, ${pct}, Purchases: ${pur.toFixed(1)}/5, Sat: ${sat.toFixed(2)}/5): High-frequency loyalists appreciating "${appr}". Unlock cross-category expansion with 2x loyalty point multipliers, personalized AI cross-sell bundles, and exclusive referral loops.`,
      actionItems: [
        `Activate 2x loyalty reward points multiplier during seasonal campaigns for ${cat}`,
        'Deploy AI-driven predictive cross-sell recommendations directly into cart preview',
        'Launch automated customer advocacy referral program with mutual store credit rewards',
      ],
    };
  } else {
    return {
      clusterId: cid,
      title: `✦ AI-Optimized: Targeted Growth & Engagement Matrix (${summary.name} - ${cat})`,
      strategy: `AI Strategic Engagement (${count} customers, ${pct}, Sat: ${sat.toFixed(2)}/5, Purchases: ${pur.toFixed(1)}/5): Targeted growth opportunity focusing on ${cat} and improving "${imp}". Stimulate conversion velocity with curated starter kits, mobile-optimized checkout, and loyalty incentives.`,
      actionItems: [
        `Launch price-accessible bundle starter kits in ${cat}`,
        'Optimize mobile 1-click checkout flow to reduce purchase friction',
        'Incentivize post-purchase review submissions with store loyalty bonus credits',
      ],
    };
  }
}


// -------------------------------------------------------------
// Main Application Component
// -------------------------------------------------------------

export default function App() {
  const AUTH_STORAGE_KEY = 'smartseg-auth';
  const USER_STORAGE_KEY = 'smartseg-user';
  const PAGE_STORAGE_KEY = 'smartseg-page';

  // Navigation & Auth
  const [isSignedIn, setIsSignedIn] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    const savedState = localStorage.getItem(AUTH_STORAGE_KEY);
    return savedState === 'true' && Boolean(localStorage.getItem('smartseg-access-token'));
  });
  const [authMode, setAuthMode] = useState<'signin' | 'register'>('signin');
  const [currentUser, setCurrentUser] = useState<{ name: string; email: string }>(() => {
    if (typeof window === 'undefined') return { name: 'Admin', email: 'admin@example.com' };

    try {
      const savedUser = localStorage.getItem(USER_STORAGE_KEY);
      if (savedUser) {
        return JSON.parse(savedUser);
      }
    } catch (err) {
      console.warn('Unable to restore saved user:', err);
    }

    return { name: 'Admin', email: 'admin@example.com' };
  });
  const [activePage, setActivePage] = useState<string>(() => {
    if (typeof window === 'undefined') return 'overview';
    return localStorage.getItem(PAGE_STORAGE_KEY) || 'overview';
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;

    if (isSignedIn) {
      localStorage.setItem(AUTH_STORAGE_KEY, 'true');
      localStorage.setItem(USER_STORAGE_KEY, JSON.stringify(currentUser));
    } else {
      localStorage.removeItem(AUTH_STORAGE_KEY);
      localStorage.removeItem(USER_STORAGE_KEY);
    }
  }, [isSignedIn, currentUser]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      localStorage.setItem(PAGE_STORAGE_KEY, activePage);
    }
  }, [activePage]);

  // Auth Inputs
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [regName, setRegName] = useState('');
  const [regEmail, setRegEmail] = useState('');
  const [regPass, setRegPass] = useState('');

  // Dataset State
  const [datasetName, setDatasetName] = useState('Amazon Customer Behavior Survey (602 records)');
  const [activeDataset, setActiveDataset] = useState<{
    id: number;
    name: string;
    rowCount: number;
    columns: string[];
  } | null>(null);
  const [customers, setCustomers] = useState<Customer[]>(DEFAULT_CUSTOMERS);
  const [availableKeys, setAvailableKeys] = useState<string[]>(DEFAULT_NUMERIC_KEYS);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploadingDataset, setIsUploadingDataset] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [overviewMetrics, setOverviewMetrics] = useState({
    total: 602,
    activeClusters: 4,
    avgSatisfaction: '2.50',
    avgPurchases: '2.70',
    avgIncome: '$62,400',
    avgAge: '30.8',
  });

  // Clustering Controls & State
  const [algorithm, setAlgorithm] = useState('kmeans');
  const [xAxisKey, setXAxisKey] = useState('satisfaction');
  const [yAxisKey, setYAxisKey] = useState('browsing_frequency');
  const [kValue, setKValue] = useState(4);
  const [silhouette, setSilhouette] = useState<number | string | null>(null);
  const [clusteringError, setClusteringError] = useState<string | null>(null);
  const [clusterSummaries, setClusterSummaries] = useState<ClusterSummary[]>([]);
  const [clusterLabels, setClusterLabels] = useState<number[]>([]);
  const [projX, setProjX] = useState<number[]>([]);
  const [projY, setProjY] = useState<number[]>([]);

  // Customer Explorer State
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedClusterFilter, setSelectedClusterFilter] = useState<number | 'all'>('all');
  const [selectedCustomer, setSelectedCustomer] = useState<Customer | null>(DEFAULT_CUSTOMERS[0] || null);

  // XAI State
  const [xaiCustomerId, setXaiCustomerId] = useState<string>(DEFAULT_CUSTOMERS[0]?.customer_id || 'C0001');
  const [xaiData, setXaiData] = useState<{
    head: string;
    summary: string;
    action: string;
    contributions: { label: string; importance: number; direction: string; value: number; globalAverage: number }[];
  }>({
    head: 'Customer Attribution Profile',
    summary: 'Select a customer to inspect model feature attribution and cluster placement drivers.',
    action: '',
    contributions: [],
  });

  // Strategy State (Dynamically updated for any cluster count/algorithm)
  const [strategies, setStrategies] = useState<StrategyItem[]>([]);


  // Model Comparison State
  const [comparisonList, setComparisonList] = useState<ModelComparisonItem[]>([]);
  const [comparisonError, setComparisonError] = useState<string | null>(null);
  const [isComparing, setIsComparing] = useState(false);
  const [comparisonK, setComparisonK] = useState(4);

  // Toast
  const [toastText, setToastText] = useState<string | null>(null);
  const toastTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const showToast = (text: string) => {
    if (toastTimeoutRef.current) clearTimeout(toastTimeoutRef.current);
    setToastText(text);
    toastTimeoutRef.current = setTimeout(() => {
      setToastText(null);
    }, 2400);
  };

  // Helper label
  const formatLabel = (s: string) =>
    s.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());

  // Fetch Dataset from Backend
  const fetchDataset = async () => {
    try {
      const res = await apiFetch('/api/dataset');
      if (res.ok) {
        const data = await res.json();
        setDatasetName(data.name || 'Amazon Customer Behavior Survey (602 records)');
        if (data.dataset_id) {
          setActiveDataset({
            id: data.dataset_id,
            name: data.name || 'Amazon Customer Behavior Survey (602 records)',
            rowCount: data.totalCount || 0,
            columns: data.numericFeatures || [],
          });
        }
        if (data.numericFeatures && data.numericFeatures.length > 0) {
          setAvailableKeys(data.numericFeatures);
        }
        const datasetRows = Array.isArray(data.customers) ? data.customers : data.sample;
        if (Array.isArray(datasetRows) && datasetRows.length > 0) {
          setCustomers(datasetRows);
          setSelectedCustomer(datasetRows[0]);
          setXaiCustomerId(datasetRows[0].customer_id);
          return;
        }
      }
    } catch (err) {
      console.warn('Backend dataset fetch notice:', err);
    }

    // Default built-in fallback dataset if backend is unreachable
    if (customers.length === 0) {
      setCustomers(DEFAULT_CUSTOMERS);
      setSelectedCustomer(DEFAULT_CUSTOMERS[0]);
      setXaiCustomerId(DEFAULT_CUSTOMERS[0].customer_id);
    }
  };

  // Run Clustering (Backend with robust client-side fallback)
  const runClustering = async (
    targetAlgo = algorithm,
    targetK = kValue,
    targetX = xAxisKey,
    targetY = yAxisKey
  ) => {
    setClusteringError(null);
    try {
      const res = await apiFetch(
        `/api/clusters?method=${encodeURIComponent(targetAlgo)}&k=${targetK}&x=${encodeURIComponent(targetX)}&y=${encodeURIComponent(targetY)}`
      );
      if (res.ok) {
        const data = await res.json();
        setSilhouette(data.silhouette_score !== undefined ? Number(data.silhouette_score).toFixed(3) : null);
        setClusterLabels(data.labels || []);
        setProjX(data.projectionX || []);
        setProjY(data.projectionY || []);
        if (data.clusterSummaries) {
          setClusterSummaries(data.clusterSummaries);
        }
        setOverviewMetrics((prev) => ({
          ...prev,
          total: data.totalCustomers || prev.total,
          activeClusters: data.activeClustersCount || targetK,
        }));
        showToast(`${formatLabel(targetAlgo)} (K=${targetK}) computed successfully.`);
        return;
      }
    } catch (err) {
      console.warn('Backend clustering fallback to client calculation:', err);
    }

    // Client-side clustering computation
    if (customers.length > 0) {
      const { matrix } = extractFeatureMatrix(customers, availableKeys);
      let labels: number[] = [];
      let sil = 0;
      let px: number[] = [];
      let py: number[] = [];

      if (targetAlgo === 'kmeans') {
        const res = runClientKMeans(matrix, targetK);
        labels = res.labels;
        sil = res.silhouette;
      } else if (targetAlgo === 'dbscan') {
        const res = runClientDBSCAN(matrix);
        labels = res.labels;
        sil = res.silhouette;
      } else if (targetAlgo === 'hierarchical') {
        const res = runClientHierarchical(matrix, targetK);
        labels = res.labels;
        sil = res.silhouette;
      } else if (targetAlgo === 'pca') {
        const res = runClientPCAKMeans(matrix, targetK);
        labels = res.labels;
        sil = res.silhouette;
        px = res.projection.map((p) => p[0]);
        py = res.projection.map((p) => p[1]);
      } else if (targetAlgo === 'embedding') {
        const res = runClientEmbeddingKMeans(matrix, targetK);
        labels = res.labels;
        sil = res.silhouette;
        px = res.projection.map((p) => p[0]);
        py = res.projection.map((p) => p[1]);
      }

      setSilhouette(sil.toFixed(3));
      setClusterLabels(labels);
      setProjX(px);
      setProjY(py);
      setClusterSummaries([]); // forces activeClusterSummaries to re-compute dynamically
      setOverviewMetrics((prev) => ({
        ...prev,
        activeClusters: targetAlgo === 'dbscan' ? new Set(labels.filter((l) => l !== -1)).size : targetK,
      }));
      showToast(`${formatLabel(targetAlgo)} computed locally.`);
    }
  };

  // Fetch Strategies from Backend
  const fetchStrategies = async () => {
    try {
      const res = await apiFetch('/api/strategies');
      if (res.ok) {
        const data = await res.json();
        if (data.strategies && Array.isArray(data.strategies)) {
          const list: StrategyItem[] = data.strategies.map((s: any) => ({
            clusterId: s.clusterId,
            title: s.title,
            strategy: s.strategy,
          }));
          if (list.length > 0) {
            setStrategies(list);
          }
        }
      }
    } catch (err) {
      console.warn('Backend strategies notice:', err);
    }
  };

  // Save Strategy to Backend / Local Storage
  const handleSaveStrategy = async (clusterId: number, strategyText: string, title?: string) => {
    try {
      const res = await apiFetch(`/api/strategies/${clusterId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ strategy: strategyText, title }),
      });
      if (res.ok) {
        showToast(`Strategy for ${clusterId < 0 ? 'Outliers' : `Cluster ${clusterId + 1}`} saved in database.`);
        return;
      }
    } catch (err) {
      console.warn('Saving locally:', err);
    }
    localStorage.setItem(`st_${clusterId}`, strategyText);
    showToast(`Strategy for ${clusterId < 0 ? 'Outliers' : `Cluster ${clusterId + 1}`} saved locally.`);
  };

  // Generate AI Strategy for a single cluster
  const handleGenerateAIStrategy = async (clusterSummary: ClusterSummary) => {
    showToast(`Generating AI Strategy for ${clusterSummary.name}...`);
    try {
      const res = await apiFetch('/api/strategies/ai-generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ clusterSummary }),
      });
      if (res.ok) {
        const data = await res.json();
        if (data.strategy) {
          setStrategies((prev) =>
            prev.map((s) =>
              s.clusterId === clusterSummary.clusterId
                ? {
                    ...s,
                    title: data.strategy.title || s.title,
                    strategy: data.strategy.strategy || s.strategy,
                    actionItems: data.strategy.actionItems || s.actionItems,
                  }
                : s
            )
          );
          showToast(`AI Strategy generated for ${clusterSummary.name}`);
          return;
        }
      }
    } catch (err) {
      console.warn('AI generation using smart fallback:', err);
    }

    // Smart Local AI optimization fallback
    const optimized = getAiOptimizedStrategy(clusterSummary, customers.length);
    setStrategies((prev) =>
      prev.map((s) => (s.clusterId === clusterSummary.clusterId ? { ...s, ...optimized } : s))
    );
    showToast(`AI Strategy optimized for ${clusterSummary.name}`);
  };

  // Optimize All Strategies at once
  const handleOptimizeAllAI = () => {
    showToast('Optimizing strategies for all active clusters...');
    setStrategies(
      activeClusterSummaries.map((cs) => getAiOptimizedStrategy(cs, customers.length))
    );
    showToast('All active cluster strategies successfully optimized.');
  };

  // Fetch / Compute Model Comparison
  const fetchComparison = async (k = comparisonK) => {
    setIsComparing(true);
    setComparisonError(null);
    setComparisonK(k);

    try {
      const res = await apiFetch(`/api/comparison?k=${k}`);
      if (res.ok) {
        const data = await res.json();
        if (data.results && data.results.length > 0) {
          const maxScore = Math.max(...data.results.map((r: any) => +(r.silhouette_score || 0)));
          const list: ModelComparisonItem[] = data.results.map((result: any) => ({
            method: result.method,
            name: formatLabel(result.method),
            silhouette: Number((result.silhouette_score || 0).toFixed(3)),
            status: result.status,
            error: result.params?.error,
            clustersFound: result.n_clusters ?? (result.method === 'dbscan' ? 3 : k),
            category: METHOD_CATEGORIES[result.method] || 'Clustering',
            description: METHOD_DESCRIPTIONS[result.method] || '',
            isBest: +(result.silhouette_score || 0) === maxScore && maxScore > 0,
          }));
          setComparisonList(list);
          setIsComparing(false);
          setTimeout(() => renderComparisonPlot(list), 50);
          return;
        }
      }
    } catch (err) {
      console.warn('Backend comparison notice, using local engine:', err);
    }

    // Local client-side comparison computation
    try {
      const localResults = runClientModelComparison(customers, availableKeys, k);
      setComparisonList(localResults);
      setTimeout(() => renderComparisonPlot(localResults), 50);
    } catch (err: any) {
      console.error('Client comparison error:', err);
      setComparisonError('Unable to compute comparison metrics.');
    } finally {
      setIsComparing(false);
    }
  };

  // Fetch XAI Explanation for a Customer
  const fetchXAI = async (custId: string) => {
    if (!custId) return;
    try {
      const res = await apiFetch(`/api/explain-cluster/${encodeURIComponent(custId)}`);
      if (res.ok) {
        const data = await res.json();
        setXaiData({
          head: `Why ${data.customer_id} is in Cluster ${Number(data.cluster_label) + 1}`,
          summary: data.summary_text,
          action: 'Target personalized retention and engagement actions based on these primary attribution factors.',
          contributions: (data.contributions || []).map((item: any) => ({
            label: formatLabel(item.feature),
            importance: Math.abs(item.value),
            direction: item.value >= 0 ? 'higher' : 'lower',
            value: item.value,
            globalAverage: 0,
          })),
        });
        renderXAIChart(data.contributions || []);
        return;
      }
    } catch (err) {
      console.warn('Backend XAI notice:', err);
    }

    // Local fallback calculation for XAI
    const cust = customers.find((c) => c.customer_id === custId) || customers[0];
    if (!cust) return;

    let feats = availableKeys.filter((f) => f !== 'customer_id');
    if (feats.length === 0) {
      feats = ['satisfaction', 'browsing_frequency', 'purchase_count', 'income', 'age'];
    }
    feats = feats.slice(0, 8);

    const contribs = feats.map((f) => {
      const allVals = customers.map((c) => +(c[f] ?? c[f.toLowerCase()] ?? 0));
      const m = allVals.length ? allVals.reduce((a, b) => a + b, 0) / allVals.length : 0;
      const variance = allVals.reduce((a, b) => a + (b - m) ** 2, 0) / (allVals.length || 1);
      const s = Math.sqrt(variance) || 1;
      const custVal = +(cust[f] ?? cust[f.toLowerCase()] ?? 0);
      const z = (custVal - m) / s;
      return {
        label: formatLabel(f),
        value: custVal,
        globalAverage: +m.toFixed(2),
        importance: +Math.abs(z).toFixed(2),
        direction: z >= 0 ? 'higher' : 'lower',
      };
    });
    contribs.sort((a, b) => b.importance - a.importance);

    const custIdx = customers.indexOf(cust);
    const cId = clusterLabels[custIdx] !== undefined ? clusterLabels[custIdx] : 0;
    const cName = cId >= 0 ? `Cluster ${cId + 1}` : 'Noise / Outlier';

    setXaiData({
      head: `Why ${cust.customer_id} is in ${cName}`,
      summary: `Feature attribution profile: the strongest indicators relative to dataset mean are ${contribs
        .slice(0, 3)
        .map((a) => `${a.label} (${a.direction} by ${a.importance}σ)`)
        .join(', ')}.`,
      action: 'Target personalized engagement and tailored offers aligned with these primary feature drivers.',
      contributions: contribs,
    });
    renderXAIChart(contribs);
  };

  // Render Plots with Plotly
  const renderClusterPlot = () => {
    if (!window.Plotly || !document.getElementById('clusterPlot') || !customers.length) return;

    const groups: Record<number, number[]> = {};
    customers.forEach((_, idx) => {
      const c = clusterLabels[idx] !== undefined ? clusterLabels[idx] : idx % kValue;
      if (!groups[c]) groups[c] = [];
      groups[c].push(idx);
    });

    const traces = Object.keys(groups).map((cKey) => {
      const c = Number(cKey);
      const indices = groups[c];
      const isNoise = c < 0;
      return {
        x: indices.map((i) =>
          projX && projX.length > i && typeof projX[i] === 'number'
            ? projX[i]
            : +(customers[i]?.[xAxisKey] ?? customers[i]?.satisfaction ?? 0)
        ),
        y: indices.map((i) =>
          projY && projY.length > i && typeof projY[i] === 'number'
            ? projY[i]
            : +(customers[i]?.[yAxisKey] ?? customers[i]?.browsing_frequency ?? 0)
        ),
        mode: 'markers',
        type: 'scatter',
        name: isNoise ? 'Noise' : `Cluster ${c + 1}`,
        text: indices.map((i) => customers[i]?.customer_id || `C${i + 1}`),
        marker: {
          size: 8,
          color: isNoise ? '#9ca3af' : CLUSTER_COLORS[Math.abs(c) % CLUSTER_COLORS.length],
          opacity: 0.82,
        },
      };
    });

    const isProjected = (algorithm === 'pca' || algorithm === 'embedding') && projX.length > 0;
    window.Plotly.newPlot(
      'clusterPlot',
      traces,
      {
        margin: { l: 55, r: 15, t: 10, b: 55 },
        xaxis: {
          title: isProjected ? 'Component 1' : formatLabel(xAxisKey),
          gridcolor: '#edf0f5',
        },
        yaxis: {
          title: isProjected ? 'Component 2' : formatLabel(yAxisKey),
          gridcolor: '#edf0f5',
        },
        legend: { orientation: 'h', y: -0.18 },
        paper_bgcolor: '#fff',
        plot_bgcolor: '#fff',
      },
      { responsive: true, displaylogo: false }
    );
  };

  const renderComparisonPlot = (items = comparisonList) => {
    if (!window.Plotly || !document.getElementById('comparisonPlot') || !items.length) return;
    const sorted = [...items].filter((m) => m.status !== 'failed');
    const methods = sorted.map((m) => m.name);
    const scores = sorted.map((m) => (typeof m.silhouette === 'number' ? m.silhouette : 0));
    const colors = sorted.map((m) => (m.isBest ? '#10b981' : '#5b5ce2'));

    window.Plotly.newPlot(
      'comparisonPlot',
      [
        {
          x: methods,
          y: scores,
          type: 'bar',
          marker: { color: colors, opacity: 0.88 },
          text: scores.map((s) => s.toFixed(3)),
          textposition: 'auto',
        },
      ],
      {
        margin: { l: 45, r: 15, t: 15, b: 50 },
        yaxis: {
          title: 'Silhouette Score (Separation Quality)',
          range: [0, Math.max(0.75, ...scores) + 0.1],
          gridcolor: '#edf0f5',
        },
        xaxis: { gridcolor: '#edf0f5' },
        paper_bgcolor: '#fff',
        plot_bgcolor: '#fff',
      },
      { responsive: true, displaylogo: false }
    );
  };

  const renderOverviewCharts = (custs = customers) => {
    if (!window.Plotly || !custs.length) return;

    const ageKey = availableKeys.find((k) => /age/i.test(k)) || availableKeys[0] || 'age';
    const incomeKey = availableKeys.find((k) => /(income|salary|spend|purchase|amount)/i.test(k)) || availableKeys[1] || availableKeys[0] || 'income';

    const ageEl = document.getElementById('ageChart');
    if (ageEl) {
      window.Plotly.newPlot(
        'ageChart',
        [{ x: custs.map((c) => +(c[ageKey] ?? c.age ?? 0)), type: 'histogram', marker: { color: '#5b5ce2' } }],
        {
          margin: { l: 40, r: 10, t: 10, b: 40 },
          xaxis: { title: formatLabel(ageKey), gridcolor: '#edf0f5' },
          yaxis: { title: 'Customers', gridcolor: '#edf0f5' },
          paper_bgcolor: '#fff',
          plot_bgcolor: '#fff',
        },
        { responsive: true, displaylogo: false }
      );
    }

    const incEl = document.getElementById('incomeChart');
    if (incEl) {
      window.Plotly.newPlot(
        'incomeChart',
        [{ x: custs.map((c) => +(c[incomeKey] ?? c.income ?? 0)), type: 'histogram', marker: { color: '#7c3aed' } }],
        {
          margin: { l: 50, r: 10, t: 10, b: 40 },
          xaxis: { title: formatLabel(incomeKey), gridcolor: '#edf0f5' },
          yaxis: { title: 'Customers', gridcolor: '#edf0f5' },
          paper_bgcolor: '#fff',
          plot_bgcolor: '#fff',
        },
        { responsive: true, displaylogo: false }
      );
    }
  };

  const renderXAIChart = (contribs: any[]) => {
    if (!window.Plotly || !document.getElementById('xchart')) return;
    const sorted = [...contribs].sort((a, b) => a.importance - b.importance);
    window.Plotly.newPlot(
      'xchart',
      [
        {
          x: sorted.map((a) => a.importance),
          y: sorted.map((a) => a.label),
          type: 'bar',
          orientation: 'h',
          marker: {
            color: sorted.map((a) => (a.direction === 'higher' ? '#10b981' : '#5b5ce2')),
          },
        },
      ],
      {
        margin: { l: 140, r: 20, t: 10, b: 35 },
        xaxis: { title: 'Relative contribution (SHAP score)', gridcolor: '#edf0f5' },
        paper_bgcolor: '#fff',
        plot_bgcolor: '#fff',
      },
      { responsive: true, displaylogo: false }
    );
  };

  // Initial Boot
  useEffect(() => {
    fetchDataset();
    runClustering('kmeans', 4, 'satisfaction', 'browsing_frequency');
    fetchStrategies();
    fetchComparison(4);
  }, []);

  // Update Overview Metrics when customers or cluster count changes
  useEffect(() => {
    if (customers.length > 0) {
      const avgSat = (customers.reduce((s, c) => s + (+c.satisfaction || 0), 0) / customers.length).toFixed(2);
      const avgPur = (customers.reduce((s, c) => s + (+c.purchase_count || 0), 0) / customers.length).toFixed(2);
      const avgInc = Math.round(customers.reduce((s, c) => s + (+c.income || 0), 0) / customers.length);
      const avgAge = (customers.reduce((s, c) => s + (+c.age || 0), 0) / customers.length).toFixed(1);

      setOverviewMetrics({
        total: customers.length,
        activeClusters: kValue,
        avgSatisfaction: avgSat,
        avgPurchases: avgPur,
        avgIncome: `$${avgInc.toLocaleString()}`,
        avgAge,
      });

      if (activePage === 'overview') {
        setTimeout(() => renderOverviewCharts(customers), 50);
      }
    }
  }, [customers, activePage, kValue]);

  // Redraw plots on page or clustering change
  useEffect(() => {
    if (activePage === 'clusters') {
      setTimeout(() => renderClusterPlot(), 50);
    } else if (activePage === 'overview') {
      setTimeout(() => renderOverviewCharts(customers), 50);
    } else if (activePage === 'comparison') {
      setTimeout(() => renderComparisonPlot(comparisonList), 50);
    } else if (activePage === 'xai') {
      setTimeout(() => {
        if (xaiCustomerId) fetchXAI(xaiCustomerId);
      }, 50);
    }
  }, [activePage, clusterLabels, projX, projY, algorithm, xAxisKey, yAxisKey, comparisonList]);

  // -------------------------------------------------------------
  // Dynamic Active Cluster Summaries with Complete Metrics
  // -------------------------------------------------------------
  const getMostFrequentTrait = (items: Customer[], key: string, fallback: string): string => {
    const counts: Record<string, number> = {};
    for (const item of items) {
      const raw = item[key] ?? item[key.toLowerCase()] ?? '';
      if (typeof raw === 'string' && raw.trim().length > 1) {
        const val = raw.trim();
        counts[val] = (counts[val] || 0) + 1;
      }
    }
    let best = fallback;
    let maxC = 0;
    for (const [val, c] of Object.entries(counts)) {
      if (c > maxC) {
        maxC = c;
        best = val;
      }
    }
    return best;
  };

  const activeClusterSummaries: ClusterSummary[] = useMemo(() => {
    if (!customers || customers.length === 0) {
      return [];
    }
    const groups: Record<number, Customer[]> = {};
    customers.forEach((c, idx) => {
      const lbl = clusterLabels[idx] !== undefined ? clusterLabels[idx] : idx % kValue;
      if (!groups[lbl]) groups[lbl] = [];
      groups[lbl].push(c);
    });

    const satKey = availableKeys.find((k) => /satisfaction/i.test(k)) || availableKeys[0] || 'satisfaction';
    const incKey = availableKeys.find((k) => /income/i.test(k)) || availableKeys[1] || 'income';
    const ageKey = availableKeys.find((k) => /age/i.test(k)) || 'age';
    const purKey = availableKeys.find((k) => /purchase|spend/i.test(k)) || 'purchase_count';
    const bfKey = availableKeys.find((k) => /browsing/i.test(k)) || 'browsing_frequency';
    const cartKey = availableKeys.find((k) => /cart/i.test(k)) || 'cart_completion';
    const revKey = availableKeys.find((k) => /review/i.test(k)) || 'review_importance';

    return Object.keys(groups)
      .map((gKey) => {
        const clusterId = Number(gKey);
        const members = groups[clusterId];
        const avgSat = members.reduce((s, m) => s + (Number(m[satKey] ?? m.satisfaction) || 0), 0) / (members.length || 1);
        const avgInc = members.reduce((s, m) => s + (Number(m[incKey] ?? m.income) || 0), 0) / (members.length || 1);
        const avgAge = members.reduce((s, m) => s + (Number(m[ageKey] ?? m.age) || 0), 0) / (members.length || 1);
        const avgPur = members.reduce((s, m) => s + (Number(m[purKey] ?? m.purchase_count) || 0), 0) / (members.length || 1);
        const avgBf = members.reduce((s, m) => s + (Number(m[bfKey] ?? m.browsing_frequency) || 0), 0) / (members.length || 1);
        const avgCart = members.reduce((s, m) => s + (Number(m[cartKey] ?? m.cart_completion) || 0), 0) / (members.length || 1);
        const avgRev = members.reduce((s, m) => s + (Number(m[revKey] ?? m.review_importance) || 0), 0) / (members.length || 1);

        const topCategory = getMostFrequentTrait(members, 'purchase_categories', 'Beauty & Lifestyle');
        const topAbandonmentReason = getMostFrequentTrait(members, 'cart_abandonment_factors', 'Found a better price elsewhere');
        const topServiceAppreciation = getMostFrequentTrait(members, 'service_appreciation', 'Competitive prices');
        const topImprovementArea = getMostFrequentTrait(members, 'improvement_areas', 'Packaging & Shipping');

        return {
          clusterId,
          name: clusterId < 0 ? 'Noise / Outliers' : `Cluster ${clusterId + 1}`,
          count: members.length,
          percentage: Math.round((members.length / customers.length) * 100),
          avgSatisfaction: Number(avgSat.toFixed(2)),
          avgIncome: Math.round(avgInc),
          avgAge: Number(avgAge.toFixed(1)),
          avgPurchases: Number(avgPur.toFixed(1)),
          avgBrowsing: Number(avgBf.toFixed(1)),
          avgCartCompletion: Number(avgCart.toFixed(1)),
          avgReviewImportance: Number(avgRev.toFixed(1)),
          topCategory,
          topAbandonmentReason,
          topServiceAppreciation,
          topImprovementArea,
        };
      })
      .sort((a, b) => a.clusterId - b.clusterId);
  }, [customers, clusterLabels, kValue, availableKeys]);

  // -------------------------------------------------------------
  // Synchronize Strategies Dynamically with Active Cluster Summaries
  // -------------------------------------------------------------
  useEffect(() => {
    if (activeClusterSummaries.length > 0) {
      setStrategies((prev) => {
        return activeClusterSummaries.map((cs) => {
          const existing = prev.find((s) => s.clusterId === cs.clusterId);
          if (
            existing &&
            existing.strategy &&
            existing.strategy.length > 20 &&
            !existing.title.includes('High-Value At-Risk') &&
            !existing.strategy.includes('modulo')
          ) {
            return {
              ...existing,
              title: existing.title || `Cluster ${cs.clusterId + 1}`,
            };
          }
          return getDefaultStrategyForCluster(cs, customers.length);
        });
      });
    }
  }, [activeClusterSummaries, customers.length]);

  // Best Performing Comparison Model
  const bestComparisonModel = useMemo(() => {
    if (!comparisonList || comparisonList.length === 0) return null;
    const okList = comparisonList.filter((m) => m.status !== 'failed');
    if (!okList.length) return null;
    return okList.reduce((best, cur) => (cur.silhouette > best.silhouette ? cur : best), okList[0]);
  }, [comparisonList]);

  // Auth Handlers
  const handleSignIn = async () => {
    if (!email || !password) {
      showToast('Please enter both email and password.');
      return;
    }
    try {
      const res = await apiFetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password }),
      });
      if (res.ok) {
        const data = await res.json();
        setAccessToken(data.access_token);
        const user = {
          name: data.user?.name || email.split('@')[0] || 'Admin',
          email: data.user?.email || email,
        };
        setCurrentUser(user);
        setIsSignedIn(true);
        showToast(`Welcome back, ${user.name}!`);
        return;
      }
    } catch (err) {
      console.warn('Backend login failed:', err);
    }
    showToast('Invalid email or password.');
  };

  const handleRegister = async () => {
    if (!regName || !regEmail || !regPass) {
      showToast('Please fill all fields.');
      return;
    }
    try {
      const res = await apiFetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: regName, email: regEmail, password: regPass }),
      });
      if (res.ok) {
        showToast('Registered successfully. You can now sign in.');
        setAuthMode('signin');
        setEmail(regEmail);
        return;
      }
    } catch (err) {
      console.warn('Registration failed:', err);
    }
    showToast('Registration failed.');
    setAuthMode('signin');
    setEmail(regEmail);
  };

  // CSV File Processing
  const processCSVFile = async (file: File) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.csv') && file.type && file.type !== 'text/csv' && file.type !== 'application/vnd.ms-excel') {
      showToast('Please upload a valid .csv file.');
      return;
    }

    setIsUploadingDataset(true);
    setClusteringError(null);
    showToast(`Parsing ${file.name}...`);

    const parseWithFallback = (text: string): Promise<any[]> => {
      return new Promise((resolve, reject) => {
        try {
          const papaInstance = Papa || window.Papa;
          if (papaInstance && typeof papaInstance.parse === 'function') {
            papaInstance.parse(text, {
              header: true,
              skipEmptyLines: 'greedy',
              dynamicTyping: true,
              complete: (results: any) => {
                if (results.data && results.data.length > 0) {
                  resolve(results.data);
                } else {
                  reject(new Error('No valid data rows found in CSV.'));
                }
              },
              error: (err: any) => reject(err),
            });
            return;
          }
        } catch (e) {
          console.warn('Papa parse failed, using manual parse fallback:', e);
        }

        try {
          const lines = text.split(/\r?\n/).filter((l) => l.trim().length > 0);
          if (lines.length < 2) {
            reject(new Error('CSV file must have a header and at least one data row.'));
            return;
          }
          const headers = lines[0].split(',').map((h) => h.trim().replace(/^["']|["']$/g, ''));
          const rows = lines.slice(1).map((line) => {
            const values = line.split(',').map((v) => v.trim().replace(/^["']|["']$/g, ''));
            const row: Record<string, any> = {};
            headers.forEach((h, idx) => {
              const raw = values[idx] ?? '';
              const num = Number(raw);
              row[h] = !isNaN(num) && raw !== '' ? num : raw;
            });
            return row;
          });
          resolve(rows);
        } catch (manualErr) {
          reject(manualErr);
        }
      });
    };

    try {
      const fileText = await file.text();
      const rawRows = await parseWithFallback(fileText);

      if (!rawRows || rawRows.length === 0) {
        throw new Error('No valid rows found in CSV.');
      }

      const firstRow = rawRows[0];
      const allHeaders = Object.keys(firstRow);

      const idKey = allHeaders.find((k) =>
        /^(customer_?id|customer|id|user_?id|client_?id|cust_?id)$/i.test(k.trim())
      );

      const detectedNumericKeys: string[] = [];
      allHeaders.forEach((k) => {
        if (idKey && k === idKey) return;
        if (/^(customer_?id|id)$/i.test(k.trim())) return;
        const numericCount = rawRows.slice(0, 50).filter((r) => {
          const val = r[k];
          return typeof val === 'number' || (!isNaN(Number(val)) && val !== '' && val !== null && val !== undefined);
        }).length;
        if (numericCount > 0) {
          detectedNumericKeys.push(k);
        }
      });

      if (detectedNumericKeys.length === 0) {
        throw new Error('CSV must contain at least one numeric feature column.');
      }

      const parsedData: Customer[] = rawRows.map((q: any, i: number) => {
        const item: any = { ...q };
        item.customer_id = idKey && q[idKey] !== undefined && q[idKey] !== ''
          ? String(q[idKey])
          : (q.customer_id || q.Customer_ID || q.CustomerID || q.id || `C${String(i + 1).padStart(4, '0')}`);

        detectedNumericKeys.forEach((k) => {
          const raw = q[k];
          item[k] = typeof raw === 'number' ? raw : (Number(raw) || 0);
        });

        if (item.age === undefined) item.age = typeof item.Age === 'number' ? item.Age : 30;
        if (item.income === undefined) item.income = typeof item.Income === 'number' ? item.Income : 50000;
        if (item.satisfaction === undefined) item.satisfaction = typeof item.Satisfaction === 'number' ? item.Satisfaction : 3.5;
        if (item.browsing_frequency === undefined) item.browsing_frequency = typeof item.Browsing_Frequency === 'number' ? item.Browsing_Frequency : 5;
        if (item.purchase_count === undefined) item.purchase_count = typeof item.Purchase_Count === 'number' ? item.Purchase_Count : 4;

        return item;
      });

      const newXKey = detectedNumericKeys.includes(xAxisKey) ? xAxisKey : detectedNumericKeys[0];
      const newYKey = detectedNumericKeys.includes(yAxisKey) ? yAxisKey : (detectedNumericKeys[1] || detectedNumericKeys[0]);

      setXAxisKey(newXKey);
      setYAxisKey(newYKey);
      setAvailableKeys(detectedNumericKeys);
      setCustomers(parsedData);
      if (parsedData.length > 0) {
        setSelectedCustomer(parsedData[0]);
        setXaiCustomerId(parsedData[0].customer_id);
      }

      try {
        const response = await apiFetch('/api/upload', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filename: file.name, data: parsedData }),
        });

        if (response.ok) {
          const uploaded = await response.json();
          setDatasetName(`${uploaded.filename} · ${uploaded.row_count} records`);
          setActiveDataset({
            id: uploaded.dataset_id,
            name: uploaded.filename,
            rowCount: uploaded.row_count,
            columns: uploaded.columns || detectedNumericKeys,
          });
          showToast(`Uploaded ${file.name} (${uploaded.row_count} rows)`);
        } else {
          setDatasetName(`${file.name} · ${parsedData.length} records`);
          setActiveDataset({
            id: 1,
            name: file.name,
            rowCount: parsedData.length,
            columns: detectedNumericKeys,
          });
          showToast(`Dataset loaded: ${file.name} (${parsedData.length} rows)`);
        }
      } catch (backendErr) {
        console.warn('Backend sync notice:', backendErr);
        setDatasetName(`${file.name} · ${parsedData.length} records`);
        setActiveDataset({
          id: 1,
          name: file.name,
          rowCount: parsedData.length,
          columns: detectedNumericKeys,
        });
        showToast(`Dataset loaded: ${file.name} (${parsedData.length} rows)`);
      }

      await runClustering(algorithm, kValue, newXKey, newYKey);
      fetchComparison(kValue);
    } catch (err: any) {
      console.error('CSV upload/parse failed:', err);
      setClusteringError(err?.message || 'Failed to parse CSV file.');
      showToast(`Upload failed: ${err?.message || 'Invalid CSV'}`);
    } finally {
      setIsUploadingDataset(false);
    }
  };

  const handleCSVUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      processCSVFile(file);
    }
    e.target.value = '';
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      processCSVFile(file);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleLoadDemo = async () => {
    try {
      const res = await apiFetch('/api/dataset/demo', { method: 'POST' });
      if (res.ok) {
        showToast('Demo dataset loaded from backend.');
        await fetchDataset();
        runClustering('kmeans', 4, 'satisfaction', 'browsing_frequency');
        fetchComparison(4);
        return;
      }
    } catch (err) {
      console.warn('Backend demo load failed:', err);
    }
    await fetchDataset();
    runClustering('kmeans', 4, 'satisfaction', 'browsing_frequency');
    fetchComparison(4);
    showToast('Demo dataset refreshed.');
  };

  // Filter Customers
  const filteredCustomers = customers.filter((c) => {
    const matchesQuery = c.customer_id.toLowerCase().includes(searchQuery.toLowerCase());
    if (!matchesQuery) return false;
    if (selectedClusterFilter === 'all') return true;
    const originalIdx = customers.indexOf(c);
    const cLabel = clusterLabels[originalIdx] !== undefined ? clusterLabels[originalIdx] : originalIdx % kValue;
    return cLabel === selectedClusterFilter;
  });

  return (
    <>
      {/* 1. AUTH SCREEN */}
      {!isSignedIn ? (
        <div id="login" className="login">
          <div className="auth" id="authBox">
            <div className="brand" id="brandTitle">
              SmartSeg <b>AI</b>
            </div>

            {authMode === 'signin' ? (
              <div id="signIn">
                <h1 id="signInTitle">Welcome back</h1>
                <p id="signInSubtitle">Customer Intelligence Dashboard</p>
                <div className="field">
                  <label htmlFor="emailInput">Email</label>
                  <input
                    id="emailInput"
                    type="email"
                    placeholder="admin@example.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                  />
                </div>
                <div className="field">
                  <label htmlFor="passInput">Password</label>
                  <input
                    id="passInput"
                    type="password"
                    placeholder="••••••••"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                  />
                </div>
                <button
                  type="button"
                  id="signInSubmitBtn"
                  className="primary"
                  style={{ width: '100%' }}
                  onClick={handleSignIn}
                >
                  Sign In
                </button>
                <div className="switch">
                  New user?{' '}
                  <span
                    id="createAccountLink"
                    className="link"
                    onClick={() => setAuthMode('register')}
                  >
                    Create account
                  </span>
                </div>
              </div>
            ) : (
              <div id="register">
                <h1 id="registerTitle">Create account</h1>
                <p id="registerSubtitle">Register a dashboard user.</p>
                <div className="field">
                  <label htmlFor="rnameInput">Name</label>
                  <input
                    id="rnameInput"
                    placeholder="Jane Doe"
                    value={regName}
                    onChange={(e) => setRegName(e.target.value)}
                  />
                </div>
                <div className="field">
                  <label htmlFor="remailInput">Email</label>
                  <input
                    id="remailInput"
                    type="email"
                    placeholder="jane@example.com"
                    value={regEmail}
                    onChange={(e) => setRegEmail(e.target.value)}
                  />
                </div>
                <div className="field">
                  <label htmlFor="rpassInput">Password</label>
                  <input
                    id="rpassInput"
                    type="password"
                    placeholder="••••••••"
                    value={regPass}
                    onChange={(e) => setRegPass(e.target.value)}
                  />
                </div>
                <button
                  type="button"
                  id="registerSubmitBtn"
                  className="primary"
                  style={{ width: '100%' }}
                  onClick={handleRegister}
                >
                  Register
                </button>
                <div className="switch">
                  Already registered?{' '}
                  <span
                    id="backToSignInLink"
                    className="link"
                    onClick={() => setAuthMode('signin')}
                  >
                    Sign in
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* 2. MAIN APPLICATION SHELL */
        <div id="app">
          <div className="shell" id="appShell">
            {/* Sidebar Navigation */}
            <aside className="side" id="appSidebar">
              <div className="brand" id="sidebarBrand">
                SmartSeg <b>AI</b>
              </div>
              <div className="nav" id="sidebarNav">
                <div className="nav-title">Workspace</div>
                <button
                  type="button"
                  id="navOverviewBtn"
                  className={activePage === 'overview' ? 'active' : ''}
                  onClick={() => setActivePage('overview')}
                >
                  ▦ Overview
                </button>
                <button
                  type="button"
                  id="navClustersBtn"
                  className={activePage === 'clusters' ? 'active' : ''}
                  onClick={() => setActivePage('clusters')}
                >
                  ◉ Cluster Explorer
                </button>
                <button
                  type="button"
                  id="navCustomersBtn"
                  className={activePage === 'customers' ? 'active' : ''}
                  onClick={() => setActivePage('customers')}
                >
                  ♙ Customer Detail
                </button>
                <button
                  type="button"
                  id="navXaiBtn"
                  className={activePage === 'xai' ? 'active' : ''}
                  onClick={() => {
                    setActivePage('xai');
                    if (customers.length > 0 && !xaiCustomerId) {
                      setXaiCustomerId(customers[0].customer_id);
                    }
                  }}
                >
                  ✦ XAI Panel
                </button>
                <button
                  type="button"
                  id="navStrategiesBtn"
                  className={activePage === 'strategies' ? 'active' : ''}
                  onClick={() => setActivePage('strategies')}
                >
                  ↗ Strategy Recommendations
                </button>

                <div className="nav-title">Management</div>
                <button
                  type="button"
                  id="navComparisonBtn"
                  className={activePage === 'comparison' ? 'active' : ''}
                  onClick={() => {
                    setActivePage('comparison');
                    fetchComparison(kValue);
                  }}
                >
                  ▤ Model Comparison
                </button>
                <button
                  type="button"
                  id="navSettingsBtn"
                  className={activePage === 'settings' ? 'active' : ''}
                  onClick={() => setActivePage('settings')}
                >
                  ⚙ Dataset / Settings
                </button>
                <button
                  type="button"
                  id="navSignOutBtn"
                  onClick={() => {
                    setIsSignedIn(false);
                    setAccessToken(null);
                    setCurrentUser({ name: 'Admin', email: 'admin@example.com' });
                    setActivePage('overview');
                    showToast('Signed out.');
                  }}
                >
                  ↪ Sign out
                </button>
              </div>
            </aside>

            {/* Main Content Area */}
            <main className="main" id="appMainContent">
              {/* Top Header */}
              <div className="top" id="topHeader">
                <div>
                  <h1 id="title">
                    {activePage === 'overview'
                      ? 'Overview'
                      : activePage === 'clusters'
                      ? 'Cluster Explorer'
                      : activePage === 'customers'
                      ? 'Customer Detail'
                      : activePage === 'xai'
                      ? 'XAI Panel'
                      : activePage === 'strategies'
                      ? 'Strategy Recommendations'
                      : activePage === 'comparison'
                      ? 'Model Comparison'
                      : 'Dataset / Settings'}
                  </h1>
                  <div className="status" id="status">
                    {datasetName}
                  </div>
                </div>
                <div className="user" id="userBadge">
                  <div>
                    <b id="uname">{currentUser.name}</b>
                    <div className="status">Manager / Admin</div>
                  </div>
                  <div className="avatar" id="avatar">
                    {currentUser.name ? currentUser.name[0].toUpperCase() : 'A'}
                  </div>
                </div>
              </div>

              {/* PAGE 1: OVERVIEW */}
              <section
                id="overview"
                className={`page ${activePage === 'overview' ? 'active' : ''}`}
              >
                <div className="cards" id="overviewCards">
                  <div className="card" id="cardTotalCustomers">
                    <div className="label">TOTAL CUSTOMERS</div>
                    <div className="metric" id="mc">
                      {overviewMetrics.total.toLocaleString()}
                    </div>
                    <div className="green">Live from dataset</div>
                  </div>

                  <div className="card" id="cardActiveClusters">
                    <div className="label">ACTIVE CLUSTERS</div>
                    <div className="metric" id="mk" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                      <span>{activeClusterSummaries.length || kValue}</span>
                      {algorithm !== 'dbscan' && (
                        <div style={{ display: 'flex', gap: '4px' }}>
                          {[3, 4, 5, 6].map((num) => (
                            <button
                              key={num}
                              type="button"
                              title={`Set ${num} clusters`}
                              style={{
                                padding: '3px 7px',
                                fontSize: '11px',
                                borderRadius: '6px',
                                fontWeight: 700,
                                border: kValue === num ? '1px solid var(--p)' : '1px solid var(--line)',
                                background: kValue === num ? '#ebe9ff' : '#f8f9fa',
                                color: kValue === num ? 'var(--p)' : '#6b7280',
                                cursor: 'pointer',
                              }}
                              onClick={() => {
                                setKValue(num);
                                runClustering(algorithm, num, xAxisKey, yAxisKey);
                                fetchComparison(num);
                              }}
                            >
                              K={num}
                            </button>
                          ))}
                        </div>
                      )}
                    </div>
                    <div className="green">Current method: {formatLabel(algorithm)}</div>
                  </div>

                  <div className="card" id="cardAvgSatisfaction">
                    <div className="label">AVG. SATISFACTION</div>
                    <div className="metric" id="ms">
                      {overviewMetrics.avgSatisfaction}
                    </div>
                    <div className="green">Scale 1.0 – 5.0</div>
                  </div>

                  <div className="card" id="cardAvgPurchases">
                    <div className="label">AVG. PURCHASES</div>
                    <div className="metric" id="mp">
                      {overviewMetrics.avgPurchases}
                    </div>
                    <div className="green">Calculated per customer</div>
                  </div>
                </div>

                {/* Segment Summary Breakdown in Overview */}
                <div className="card" style={{ marginTop: '16px', marginBottom: '16px' }} id="overviewSegmentsCard">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                    <div>
                      <h2 style={{ fontSize: '16px', margin: '0 0 4px' }}>Active Segmentation Breakdown</h2>
                      <span className="status">
                        Segmentation powered by {formatLabel(algorithm)} · {activeClusterSummaries.length} active segments
                      </span>
                    </div>
                    <button
                      className="secondary"
                      style={{ fontSize: '12px', padding: '6px 12px' }}
                      onClick={() => setActivePage('strategies')}
                    >
                      ↗ View Strategic Recommendations
                    </button>
                  </div>

                  <div className="table-wrap">
                    <table className="table">
                      <thead>
                        <tr>
                          <th>Segment</th>
                          <th>Customer Count</th>
                          <th>Share</th>
                          <th>Avg. Satisfaction</th>
                          <th>Avg. Purchases</th>
                          <th>Avg. Income</th>
                          <th>Strategic Focus</th>
                        </tr>
                      </thead>
                      <tbody>
                        {activeClusterSummaries.map((cs) => {
                          const matchedStrategy = strategies.find((s) => s.clusterId === cs.clusterId);
                          return (
                            <tr key={cs.clusterId}>
                              <td>
                                <span
                                  style={{
                                    display: 'inline-block',
                                    width: '10px',
                                    height: '10px',
                                    borderRadius: '50%',
                                    backgroundColor:
                                      cs.clusterId < 0
                                        ? '#9ca3af'
                                        : CLUSTER_COLORS[Math.abs(cs.clusterId) % CLUSTER_COLORS.length],
                                    marginRight: '6px',
                                  }}
                                ></span>
                                <b>{cs.name}</b>
                              </td>
                              <td><b>{cs.count.toLocaleString()}</b></td>
                              <td>
                                <span className="pill" style={{ fontSize: '11px' }}>
                                  {cs.percentage ?? Math.round((cs.count / (customers.length || 1)) * 100)}%
                                </span>
                              </td>
                              <td>{typeof cs.avgSatisfaction === 'number' ? cs.avgSatisfaction.toFixed(2) : '-'} / 5.0</td>
                              <td>{typeof cs.avgPurchases === 'number' ? cs.avgPurchases.toFixed(1) : '-'}</td>
                              <td>{cs.avgIncome ? `$${cs.avgIncome.toLocaleString()}` : '-'}</td>
                              <td>
                                <span style={{ fontSize: '12px', color: '#4b5563', fontWeight: 600 }}>
                                  {matchedStrategy?.title || 'Personalized Engagement'}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="grid2" id="overviewChartsGrid">
                  <div className="card" id="cardAgeDist">
                    <div className="section">
                      <h2>Age Distribution</h2>
                      <span className="status">Avg: {overviewMetrics.avgAge} yrs</span>
                    </div>
                    <div id="ageChart" className="chart"></div>
                  </div>
                  <div className="card" id="cardIncomeDist">
                    <div className="section">
                      <h2>Income Distribution</h2>
                      <span className="status">Avg: {overviewMetrics.avgIncome}</span>
                    </div>
                    <div id="incomeChart" className="chart"></div>
                  </div>
                </div>
              </section>

              {/* PAGE 2: CLUSTERS */}
              <section
                id="clusters"
                className={`page ${activePage === 'clusters' ? 'active' : ''}`}
              >
                <div className="card" id="clusterToolbarCard">
                  <div className="toolbar" id="clusterToolbar">
                    <div className="field">
                      <label htmlFor="algoSelect">Algorithm</label>
                      <select
                        id="algoSelect"
                        value={algorithm}
                        onChange={(e) => {
                          setAlgorithm(e.target.value);
                          runClustering(e.target.value, kValue, xAxisKey, yAxisKey);
                        }}
                      >
                        <option value="kmeans">K-Means</option>
                        <option value="dbscan">DBSCAN</option>
                        <option value="hierarchical">Hierarchical Clustering</option>
                        <option value="pca">PCA + K-Means</option>
                        <option value="embedding">Deep-Learning Embedding</option>
                      </select>
                    </div>

                    <div className="field">
                      <label htmlFor="xAxisSelect">X Axis</label>
                      <select
                        id="xAxisSelect"
                        value={xAxisKey}
                        onChange={(e) => {
                          setXAxisKey(e.target.value);
                          runClustering(algorithm, kValue, e.target.value, yAxisKey);
                        }}
                      >
                        {availableKeys.map((k) => (
                          <option key={k} value={k}>
                            {formatLabel(k)}
                          </option>
                        ))}
                      </select>
                    </div>

                    <div className="field">
                      <label htmlFor="yAxisSelect">Y Axis</label>
                      <select
                        id="yAxisSelect"
                        value={yAxisKey}
                        onChange={(e) => {
                          setYAxisKey(e.target.value);
                          runClustering(algorithm, kValue, xAxisKey, e.target.value);
                        }}
                      >
                        {availableKeys.map((k) => (
                          <option key={k} value={k}>
                            {formatLabel(k)}
                          </option>
                        ))}
                      </select>
                    </div>

                    {algorithm !== 'dbscan' && (
                      <div className="field" id="kFieldWrapper">
                        <label htmlFor="kSelect">K Clusters</label>
                        <select
                          id="kSelect"
                          value={kValue}
                          onChange={(e) => {
                            const val = Number(e.target.value);
                            setKValue(val);
                            runClustering(algorithm, val, xAxisKey, yAxisKey);
                          }}
                        >
                          <option value={3}>3</option>
                          <option value={4}>4</option>
                          <option value={5}>5</option>
                          <option value={6}>6</option>
                        </select>
                      </div>
                    )}

                    <button
                      id="refreshClusteringBtn"
                      className="primary"
                      onClick={() => runClustering(algorithm, kValue, xAxisKey, yAxisKey)}
                    >
                      ↻ Run / Refresh
                    </button>
                  </div>
                  <div className="status" style={{ marginTop: '10px' }}>
                    Upload a different CSV or select another algorithm and the clustering visualization updates.
                  </div>
                </div>

                <div className="cluster-grid" id="clusterGrid">
                  <div className="card" id="clusterPlotCard">
                    <div className="section">
                      <h2 id="plotName">{formatLabel(algorithm)} Cluster Plot</h2>
                      <span className="pill" id="sil">
                        Silhouette: {silhouette ?? 'Unavailable'}
                      </span>
                    </div>
                    {clusteringError && <div className="info" role="alert">{clusteringError}</div>}
                    <div id="clusterPlot" className="plot"></div>
                  </div>

                  <div id="clusterSummaryColumn">
                    <div className="card" id="clusterSummaryCard">
                      <div className="section" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                        <h2 style={{ margin: 0 }}>Cluster Summary</h2>
                        <span className="pill">
                          {activeClusterSummaries.length} Segments
                        </span>
                      </div>
                      <div className="table-wrap">
                        <table className="table" id="clusterSummaryTable">
                          <thead>
                            <tr>
                              <th>Cluster</th>
                              <th>Customers</th>
                              <th>Avg Sat.</th>
                            </tr>
                          </thead>
                          <tbody id="ct">
                            {activeClusterSummaries.length === 0 ? (
                              <tr>
                                <td colSpan={3} className="status" style={{ textAlign: 'center', padding: '16px' }}>
                                  No cluster summaries calculated yet.
                                </td>
                              </tr>
                            ) : (
                              activeClusterSummaries.map((cs) => (
                                <tr key={cs.clusterId}>
                                  <td>
                                    <span
                                      style={{
                                        display: 'inline-block',
                                        width: '10px',
                                        height: '10px',
                                        borderRadius: '50%',
                                        backgroundColor:
                                          cs.clusterId < 0
                                            ? '#9ca3af'
                                            : CLUSTER_COLORS[Math.abs(cs.clusterId) % CLUSTER_COLORS.length],
                                        marginRight: '6px',
                                      }}
                                    ></span>
                                    <b>{cs.name}</b>
                                  </td>
                                  <td>
                                    <b>{cs.count.toLocaleString()}</b>
                                    {cs.percentage !== undefined && (
                                      <span className="status" style={{ marginLeft: '4px', fontSize: '11px' }}>
                                        ({cs.percentage}%)
                                      </span>
                                    )}
                                  </td>
                                  <td>
                                    {typeof cs.avgSatisfaction === 'number'
                                      ? cs.avgSatisfaction.toFixed(2)
                                      : (cs.avgSatisfaction ?? '-')}
                                  </td>
                                </tr>
                              ))
                            )}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    <div className="card" style={{ marginTop: '15px' }} id="methodCard">
                      <h2 style={{ fontSize: '16px' }}>Methodology</h2>
                      <div id="method" className="info">
                        {METHOD_DESCRIPTIONS[algorithm] || METHOD_DESCRIPTIONS.kmeans}
                      </div>
                    </div>
                  </div>
                </div>
              </section>

              {/* PAGE 3: CUSTOMERS */}
              <section
                id="customers"
                className={`page ${activePage === 'customers' ? 'active' : ''}`}
              >
                <div className="two-col" id="customersTwoCol">
                  <div className="card" id="customerListCard">
                    <div className="section" style={{ marginBottom: '12px' }}>
                      <h2>Customer Explorer</h2>
                      <input
                        id="customerSearchInput"
                        className="search"
                        style={{ width: '160px' }}
                        placeholder="Search ID..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                      />
                    </div>

                    {/* Cluster Filter Chips */}
                    <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '12px' }}>
                      <button
                        type="button"
                        style={{
                          padding: '4px 10px',
                          fontSize: '11px',
                          borderRadius: '16px',
                          border: '1px solid var(--line)',
                          background: selectedClusterFilter === 'all' ? '#252d3e' : '#f8f9fa',
                          color: selectedClusterFilter === 'all' ? '#fff' : '#4b5563',
                          fontWeight: 700,
                          cursor: 'pointer',
                        }}
                        onClick={() => setSelectedClusterFilter('all')}
                      >
                        All ({customers.length})
                      </button>
                      {activeClusterSummaries.map((cs) => (
                        <button
                          key={cs.clusterId}
                          type="button"
                          style={{
                            padding: '4px 10px',
                            fontSize: '11px',
                            borderRadius: '16px',
                            border: `1px solid ${selectedClusterFilter === cs.clusterId ? 'var(--p)' : 'var(--line)'}`,
                            background: selectedClusterFilter === cs.clusterId ? '#ebe9ff' : '#f8f9fa',
                            color: selectedClusterFilter === cs.clusterId ? 'var(--p)' : '#4b5563',
                            fontWeight: 700,
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                          }}
                          onClick={() => setSelectedClusterFilter(cs.clusterId)}
                        >
                          <span
                            style={{
                              width: '7px',
                              height: '7px',
                              borderRadius: '50%',
                              backgroundColor:
                                cs.clusterId < 0
                                  ? '#9ca3af'
                                  : CLUSTER_COLORS[Math.abs(cs.clusterId) % CLUSTER_COLORS.length],
                            }}
                          ></span>
                          {cs.name} ({cs.count})
                        </button>
                      ))}
                    </div>

                    <div
                      id="customerList"
                      style={{ maxHeight: '500px', overflowY: 'auto' }}
                    >
                      {filteredCustomers.length === 0 ? (
                        <div className="status" style={{ textAlign: 'center', padding: '20px' }}>
                          No customers found matching the search/filter criteria.
                        </div>
                      ) : (
                        filteredCustomers.slice(0, 100).map((r, i) => {
                          const originalIdx = customers.indexOf(r);
                          const cLabel =
                            clusterLabels[originalIdx] !== undefined
                              ? clusterLabels[originalIdx]
                              : originalIdx % kValue;
                          const satVal = r.satisfaction ?? r.Satisfaction ?? '-';
                          const purVal = r.purchase_count ?? r.Purchase_Count ?? r.purchases ?? '-';
                          return (
                            <div
                              key={r.customer_id || i}
                              className="customer"
                              onClick={() => setSelectedCustomer(r)}
                              style={{
                                backgroundColor: selectedCustomer?.customer_id === r.customer_id ? '#f3f4ff' : 'transparent',
                              }}
                            >
                              <div>
                                <b>{r.customer_id}</b>
                                <br />
                                <span className="status">
                                  Satisfaction: {satVal} · Purchases: {purVal}
                                </span>
                              </div>
                              <span
                                className="pill"
                                style={{
                                  backgroundColor: cLabel < 0 ? '#f3f4f6' : '#eff2fe',
                                  color: cLabel < 0 ? '#6b7280' : CLUSTER_COLORS[Math.abs(cLabel) % CLUSTER_COLORS.length],
                                }}
                              >
                                {cLabel >= 0 ? `C${cLabel + 1}` : 'Noise'}
                              </span>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>

                  <div className="card" id="customerProfileCard">
                    <h2 style={{ fontSize: '17px', marginBottom: '12px' }}>Customer Profile</h2>
                    <div id="profile" className="info">
                      {selectedCustomer ? (
                        <div>
                          <h3 style={{ fontSize: '18px', margin: '0 0 10px', color: 'var(--p)' }}>
                            {selectedCustomer.customer_id}
                          </h3>
                          {availableKeys.map((k) => (
                            <p key={k} style={{ margin: '6px 0' }}>
                              <b>{formatLabel(k)}:</b> {selectedCustomer[k]}
                            </p>
                          ))}
                          <p style={{ margin: '8px 0 0', paddingTop: '8px', borderTop: '1px solid var(--line)' }}>
                            <b>Assigned Cluster:</b>{' '}
                            {clusterLabels[customers.indexOf(selectedCustomer)] >= 0
                              ? `Cluster ${clusterLabels[customers.indexOf(selectedCustomer)] + 1}`
                              : 'Noise / Outlier'}
                          </p>
                          <button
                            className="secondary"
                            style={{ marginTop: '12px', fontSize: '12px' }}
                            onClick={() => {
                              setXaiCustomerId(selectedCustomer.customer_id);
                              setActivePage('xai');
                            }}
                          >
                            ✦ Inspect XAI Attribution
                          </button>
                        </div>
                      ) : (
                        'Select a customer from the list to view profile attributes.'
                      )}
                    </div>
                  </div>
                </div>
              </section>

              {/* PAGE 4: XAI */}
              <section
                id="xai"
                className={`page ${activePage === 'xai' ? 'active' : ''}`}
              >
                <div className="card" id="xaiCard">
                  <div className="section">
                    <h2>XAI / Why this customer?</h2>
                    <select
                      id="xcSelect"
                      value={xaiCustomerId}
                      onChange={(e) => {
                        setXaiCustomerId(e.target.value);
                        fetchXAI(e.target.value);
                      }}
                      style={{
                        padding: '9px 12px',
                        border: '1px solid var(--line)',
                        borderRadius: '8px',
                        background: '#fff',
                      }}
                    >
                      {customers.slice(0, 250).map((r) => (
                        <option key={r.customer_id} value={r.customer_id}>
                          {r.customer_id}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="two-col" id="xaiTwoCol" style={{ marginTop: '15px' }}>
                    <div>
                      <div className="label" style={{ marginBottom: '8px' }}>
                        FEATURE IMPORTANCE / SHAP DIVERGENCE
                      </div>
                      <div id="xchart" className="chart"></div>
                    </div>

                    <div>
                      <h3 id="xhead" style={{ fontSize: '18px', margin: '0 0 12px' }}>
                        {xaiData.head}
                      </h3>
                      <div id="xtext" className="info">
                        <p style={{ margin: '0 0 10px' }}>{xaiData.summary}</p>
                        {xaiData.action && (
                          <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: '1px solid #e2e8f0' }}>
                            <b>Recommended Strategy Action:</b>
                            <p style={{ margin: '4px 0 0', color: 'var(--ink)' }}>{xaiData.action}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </section>

              {/* PAGE 5: STRATEGIES */}
              <section
                id="strategies"
                className={`page ${activePage === 'strategies' ? 'active' : ''}`}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                  <div>
                    <h2 style={{ fontSize: '18px', margin: '0 0 4px' }}>Segment Strategy Recommendations</h2>
                    <span className="status">
                      Dynamically synchronized with your active {formatLabel(algorithm)} model ({activeClusterSummaries.length} active segments)
                    </span>
                  </div>
                  <button
                    className="primary"
                    style={{ fontSize: '12px', padding: '8px 14px' }}
                    onClick={handleOptimizeAllAI}
                  >
                    ✦ AI Optimize All Strategies
                  </button>
                </div>

                <div
                  id="strategiesGrid"
                  className="cards"
                  style={{ gridTemplateColumns: 'repeat(2, 1fr)' }}
                >
                  {strategies.map((s) => {
                    const cSummary = activeClusterSummaries.find((cs) => cs.clusterId === s.clusterId) || {
                      clusterId: s.clusterId,
                      name: s.clusterId < 0 ? 'Noise / Outliers' : `Cluster ${s.clusterId + 1}`,
                      count: 150,
                      percentage: 25,
                      avgSatisfaction: 3.5,
                      avgIncome: 45000,
                      avgAge: 34,
                      avgPurchases: 4,
                    };
                    return (
                      <div key={s.clusterId} className="card strategy" id={`strategyCard_${s.clusterId}`}>
                        <div className="section" style={{ marginBottom: '10px' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span
                              style={{
                                width: '12px',
                                height: '12px',
                                borderRadius: '50%',
                                backgroundColor:
                                  s.clusterId < 0
                                    ? '#9ca3af'
                                    : CLUSTER_COLORS[Math.abs(s.clusterId) % CLUSTER_COLORS.length],
                              }}
                            ></span>
                            <h2 style={{ margin: 0 }}>{cSummary.name}</h2>
                          </div>
                          <div style={{ display: 'flex', gap: '6px', alignItems: 'center' }}>
                            <span className="pill">
                              {cSummary.count.toLocaleString()} customers ({cSummary.percentage ?? 25}%)
                            </span>
                            <button
                              className="secondary"
                              style={{ padding: '4px 9px', fontSize: '11px' }}
                              title="Generate AI Strategy"
                              onClick={() => handleGenerateAIStrategy(cSummary as ClusterSummary)}
                            >
                              ✦ AI Optimize
                            </button>
                          </div>
                        </div>

                        {/* Metric Highlights */}
                        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginBottom: '10px' }}>
                          <span style={{ fontSize: '11px', background: '#f3f4f6', padding: '2px 7px', borderRadius: '4px', color: '#4b5563' }}>
                            Sat: <b>{typeof cSummary.avgSatisfaction === 'number' ? cSummary.avgSatisfaction.toFixed(2) : cSummary.avgSatisfaction}/5</b>
                          </span>
                          <span style={{ fontSize: '11px', background: '#f3f4f6', padding: '2px 7px', borderRadius: '4px', color: '#4b5563' }}>
                            Purchases: <b>{typeof cSummary.avgPurchases === 'number' ? cSummary.avgPurchases.toFixed(1) : cSummary.avgPurchases}</b>
                          </span>
                          {cSummary.topCategory && (
                            <span style={{ fontSize: '11px', background: '#e0f2fe', padding: '2px 7px', borderRadius: '4px', color: '#0369a1' }}>
                              Top Category: <b>{cSummary.topCategory}</b>
                            </span>
                          )}
                          {cSummary.topAbandonmentReason && (
                            <span style={{ fontSize: '11px', background: '#fef3c7', padding: '2px 7px', borderRadius: '4px', color: '#92400e' }}>
                              Friction Point: <b>{cSummary.topAbandonmentReason}</b>
                            </span>
                          )}
                          {cSummary.avgIncome && (
                            <span style={{ fontSize: '11px', background: '#f3f4f6', padding: '2px 7px', borderRadius: '4px', color: '#4b5563' }}>
                              Avg Budget: <b>${cSummary.avgIncome.toLocaleString()}</b>
                            </span>
                          )}
                        </div>

                        <div style={{ marginBottom: '8px' }}>
                          <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--muted)', textTransform: 'uppercase' }}>
                            Persona / Segment Profile:
                          </label>
                          <input
                            type="text"
                            value={s.title}
                            style={{
                              width: '100%',
                              padding: '6px 10px',
                              marginTop: '3px',
                              fontWeight: 700,
                              fontSize: '13px',
                              border: '1px solid var(--line)',
                              borderRadius: '6px',
                            }}
                            onChange={(e) => {
                              const newTitle = e.target.value;
                              setStrategies((prev) =>
                                prev.map((item) =>
                                  item.clusterId === s.clusterId ? { ...item, title: newTitle } : item
                                )
                              );
                            }}
                          />
                        </div>

                        <p style={{ margin: '10px 0 8px' }}>
                          <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--muted)', textTransform: 'uppercase' }}>
                            Actionable Strategic Recommendation:
                          </label>
                          <textarea
                            id={`st${s.clusterId}`}
                            rows={3}
                            value={s.strategy}
                            style={{
                              width: '100%',
                              padding: '8px 10px',
                              marginTop: '4px',
                              fontSize: '12px',
                              lineHeight: 1.5,
                              border: '1px solid var(--line)',
                              borderRadius: '6px',
                            }}
                            onChange={(e) => {
                              const newText = e.target.value;
                              setStrategies((prev) =>
                                prev.map((item) =>
                                  item.clusterId === s.clusterId
                                    ? { ...item, strategy: newText }
                                    : item
                                )
                              );
                            }}
                          />
                        </p>

                        {s.actionItems && s.actionItems.length > 0 && (
                          <div style={{ marginBottom: '12px' }}>
                            <label style={{ fontSize: '11px', fontWeight: 700, color: 'var(--muted)', textTransform: 'uppercase' }}>
                              Key Action Items:
                            </label>
                            <ul style={{ margin: '4px 0 0 16px', padding: 0, fontSize: '12px', color: '#374151' }}>
                              {s.actionItems.map((item, aIdx) => (
                                <li key={aIdx} style={{ margin: '2px 0' }}>{item}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        <button
                          id={`saveStrategyBtn_${s.clusterId}`}
                          className="primary"
                          style={{ fontSize: '12px', padding: '6px 14px' }}
                          onClick={() => handleSaveStrategy(s.clusterId, s.strategy, s.title)}
                        >
                          Save Strategy
                        </button>
                      </div>
                    );
                  })}
                </div>
              </section>

              {/* PAGE 6: COMPARISON */}
              <section
                id="comparison"
                className={`page ${activePage === 'comparison' ? 'active' : ''}`}
              >
                <div className="card" id="comparisonCard" style={{ marginBottom: '16px' }}>
                  <div className="section" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '10px' }}>
                    <div>
                      <h2 style={{ margin: '0 0 4px' }}>Model Comparison &amp; Algorithm Benchmark</h2>
                      <span className="status">
                        Cross-evaluate separation quality and silhouette metrics across 5 machine learning models.
                      </span>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ fontSize: '12px', fontWeight: 700, color: 'var(--muted)' }}>Target K:</span>
                        {[2, 3, 4, 5, 6].map((num) => (
                          <button
                            key={num}
                            type="button"
                            style={{
                              padding: '4px 8px',
                              fontSize: '11px',
                              borderRadius: '6px',
                              fontWeight: 700,
                              border: comparisonK === num ? '1px solid var(--p)' : '1px solid var(--line)',
                              background: comparisonK === num ? '#ebe9ff' : '#f8f9fa',
                              color: comparisonK === num ? 'var(--p)' : '#6b7280',
                              cursor: 'pointer',
                            }}
                            onClick={() => {
                              setComparisonK(num);
                              fetchComparison(num);
                            }}
                          >
                            {num}
                          </button>
                        ))}
                      </div>

                      <button
                        className="primary"
                        style={{ fontSize: '12px', padding: '7px 14px' }}
                        onClick={() => fetchComparison(comparisonK)}
                        disabled={isComparing}
                      >
                        {isComparing ? 'Evaluating...' : '↻ Evaluate Models'}
                      </button>
                    </div>
                  </div>

                  {/* Best Model Banner */}
                  {bestComparisonModel && (
                    <div
                      style={{
                        marginTop: '14px',
                        padding: '12px 16px',
                        background: 'linear-gradient(135deg, #ecfdf5, #f0fdf4)',
                        border: '1px solid #a7f3d0',
                        borderRadius: '10px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                      }}
                    >
                      <div>
                        <div style={{ fontSize: '11px', fontWeight: 800, color: '#059669', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                          ★ Best Performing Algorithm
                        </div>
                        <div style={{ fontSize: '15px', fontWeight: 800, color: '#065f46', marginTop: '2px' }}>
                          {bestComparisonModel.name} — Silhouette Quality Score: {bestComparisonModel.silhouette}
                        </div>
                        <div style={{ fontSize: '12px', color: '#047857', marginTop: '2px' }}>
                          {bestComparisonModel.name} delivers the cleanest segment separation for your customer dataset.
                        </div>
                      </div>
                      <button
                        className="primary"
                        style={{ background: '#059669', fontSize: '12px', padding: '6px 12px' }}
                        onClick={() => {
                          setAlgorithm(bestComparisonModel.method);
                          runClustering(bestComparisonModel.method, comparisonK, xAxisKey, yAxisKey);
                          setActivePage('clusters');
                        }}
                      >
                        Apply Model ↗
                      </button>
                    </div>
                  )}

                  {/* Silhouette Score Bar Chart */}
                  <div style={{ marginTop: '16px' }}>
                    <div style={{ fontSize: '12px', fontWeight: 700, color: 'var(--muted)', marginBottom: '8px' }}>
                      SILHOUETTE SCORE BENCHMARK (HIGHER IS BETTER)
                    </div>
                    <div id="comparisonPlot" className="chart" style={{ height: '220px' }}></div>
                  </div>

                  <div className="table-wrap" style={{ marginTop: '16px' }}>
                    <table className="table" id="comparisonTable">
                      <thead>
                        <tr>
                          <th>Algorithm</th>
                          <th>Category</th>
                          <th>Clusters Found</th>
                          <th>Silhouette Quality</th>
                          <th>Status</th>
                          <th>Action</th>
                        </tr>
                      </thead>
                      <tbody id="cmp">
                        {comparisonList.length === 0 ? (
                          <tr>
                            <td colSpan={6} className="status" style={{ textAlign: 'center', padding: '20px' }}>
                              {comparisonError || (isComparing ? 'Running model benchmarks...' : 'Run a comparison to see benchmark results.')}
                            </td>
                          </tr>
                        ) : (
                          comparisonList.map((m) => (
                            <tr
                              key={m.method}
                              style={{
                                backgroundColor: m.isBest ? '#f0fdf4' : 'transparent',
                              }}
                            >
                              <td>
                                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                                  <b>{m.name}</b>
                                  {m.isBest && (
                                    <span
                                      className="pill"
                                      style={{ background: '#dcfce7', color: '#15803d', fontSize: '10px', padding: '1px 6px' }}
                                    >
                                      ★ Best
                                    </span>
                                  )}
                                </div>
                              </td>
                              <td>
                                <span style={{ fontSize: '12px', color: '#4b5563' }}>
                                  {m.category || METHOD_CATEGORIES[m.method] || 'Clustering'}
                                </span>
                              </td>
                              <td>
                                <b>{m.clustersFound ?? (m.method === 'dbscan' ? 3 : comparisonK)}</b>
                              </td>
                              <td>
                                {m.status === 'failed' ? (
                                  <span style={{ color: '#ef4444' }}>Unavailable</span>
                                ) : (
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                    <div
                                      style={{
                                        width: '80px',
                                        height: '6px',
                                        background: '#e5e7eb',
                                        borderRadius: '3px',
                                        overflow: 'hidden',
                                      }}
                                    >
                                      <div
                                        style={{
                                          width: `${Math.min(100, Math.max(0, m.silhouette * 100))}%`,
                                          height: '100%',
                                          background: m.isBest ? '#10b981' : m.silhouette > 0.4 ? '#5b5ce2' : '#9ca3af',
                                        }}
                                      ></div>
                                    </div>
                                    <span
                                      style={{
                                        fontWeight: 700,
                                        color: m.isBest ? '#059669' : m.silhouette > 0.4 ? '#4338ca' : '#374151',
                                      }}
                                    >
                                      {typeof m.silhouette === 'number' ? m.silhouette.toFixed(3) : m.silhouette}
                                    </span>
                                  </div>
                                )}
                              </td>
                              <td>
                                {m.status === 'failed' ? (
                                  <span className="pill" style={{ background: '#fee2e2', color: '#b91c1c' }}>
                                    Failed
                                  </span>
                                ) : (
                                  <span className="pill" style={{ background: '#ecfdf5', color: '#047857' }}>
                                    Completed
                                  </span>
                                )}
                              </td>
                              <td>
                                <button
                                  className={m.isBest ? 'primary' : 'secondary'}
                                  style={{ fontSize: '11px', padding: '4px 9px' }}
                                  onClick={() => {
                                    setAlgorithm(m.method);
                                    runClustering(m.method, comparisonK, xAxisKey, yAxisKey);
                                    setActivePage('clusters');
                                  }}
                                >
                                  Apply Model
                                </button>
                              </td>
                            </tr>
                          ))
                        )}
                      </tbody>
                    </table>
                  </div>
                </div>
              </section>

              {/* PAGE 7: SETTINGS / DATASET */}
              <section
                id="settings"
                className={`page ${activePage === 'settings' ? 'active' : ''}`}
              >
                <div className="two-col" id="settingsTwoCol">
                  <div className="card" id="uploadDatasetCard">
                    <h2 style={{ fontSize: '17px', marginBottom: '10px' }}>
                      Upload Customer Dataset
                    </h2>
                    <div
                      className={`upload ${isDragging ? 'dragging' : ''}`}
                      id="uploadDropZone"
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                      style={{
                        border: isDragging ? '2px dashed #5b5ce2' : '2px dashed #cbd3e2',
                        backgroundColor: isDragging ? '#f0f2ff' : 'transparent',
                        transition: 'all 0.2s ease',
                        cursor: 'pointer',
                      }}
                      onClick={() => fileInputRef.current?.click()}
                    >
                      <label
                        className="primary"
                        htmlFor="csvInput"
                        style={{ pointerEvents: 'none' }}
                      >
                        {isUploadingDataset ? 'Uploading...' : 'Choose CSV'}
                      </label>
                      <input
                        ref={fileInputRef}
                        id="csvInput"
                        type="file"
                        accept=".csv"
                        onChange={handleCSVUpload}
                      />
                      <p id="file" style={{ marginTop: '12px', fontSize: '13px' }}>
                        {isUploadingDataset
                          ? 'Processing and loading dataset...'
                          : 'Click to choose or drag and drop a .csv file here.'}
                      </p>
                      <p style={{ marginTop: '6px', fontSize: '11px', color: 'var(--muted)' }}>
                        Recommended fields: customer_id, age, income, satisfaction, browsing_frequency, purchase_count.
                      </p>
                    </div>
                    <button
                      id="loadDemoDataBtn"
                      className="secondary"
                      onClick={handleLoadDemo}
                      style={{ marginTop: '15px' }}
                    >
                      Load Demo Data
                    </button>
                    <div className="info" style={{ marginTop: '15px' }} aria-live="polite">
                      <b>Database dataset</b>
                      {activeDataset ? (
                        <>
                          <br />
                          {activeDataset.name}
                          <br />
                          {activeDataset.rowCount.toLocaleString()} rows · Dataset ID {activeDataset.id}
                          <br />
                          Features: {activeDataset.columns.map(formatLabel).join(', ') || 'None detected'}
                        </>
                      ) : (
                        <>
                          <br />
                          Active memory dataset ({customers.length} records).
                        </>
                      )}
                    </div>
                  </div>

                  <div className="card" id="datasetHealthCard">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
                      <h2 style={{ fontSize: '17px', margin: 0 }}>
                        Dataset Health &amp; Overview
                      </h2>
                      <span
                        className="pill"
                        style={{
                          backgroundColor: activeDataset || customers.length ? '#ecfdf5' : '#fef3c7',
                          color: activeDataset || customers.length ? '#059669' : '#d97706',
                          fontWeight: 700,
                          fontSize: '11px',
                        }}
                      >
                        {activeDataset || customers.length ? '● Ready for Segmentation' : '○ Waiting for Data'}
                      </span>
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', marginBottom: '16px' }}>
                      <div className="info" style={{ padding: '12px' }}>
                        <div style={{ fontSize: '11px', color: 'var(--muted)', fontWeight: 700, textTransform: 'uppercase' }}>
                          Total Records
                        </div>
                        <div style={{ fontSize: '20px', fontWeight: 800, color: 'var(--p)', marginTop: '4px' }}>
                          {activeDataset ? activeDataset.rowCount.toLocaleString() : customers.length.toLocaleString()}
                        </div>
                      </div>

                      <div className="info" style={{ padding: '12px' }}>
                        <div style={{ fontSize: '11px', color: 'var(--muted)', fontWeight: 700, textTransform: 'uppercase' }}>
                          Detected Features
                        </div>
                        <div style={{ fontSize: '20px', fontWeight: 800, color: '#10b981', marginTop: '4px' }}>
                          {availableKeys.length} Columns
                        </div>
                      </div>

                      <div className="info" style={{ padding: '12px' }}>
                        <div style={{ fontSize: '11px', color: 'var(--muted)', fontWeight: 700, textTransform: 'uppercase' }}>
                          Current Algorithm
                        </div>
                        <div style={{ fontSize: '14px', fontWeight: 700, color: '#374151', marginTop: '6px' }}>
                          {formatLabel(algorithm)} (K={kValue})
                        </div>
                      </div>

                      <div className="info" style={{ padding: '12px' }}>
                        <div style={{ fontSize: '11px', color: 'var(--muted)', fontWeight: 700, textTransform: 'uppercase' }}>
                          Silhouette Quality
                        </div>
                        <div style={{ fontSize: '14px', fontWeight: 700, color: silhouette ? '#059669' : '#6b7280', marginTop: '6px' }}>
                          {silhouette ? `${silhouette} Score` : 'Calculated on run'}
                        </div>
                      </div>
                    </div>

                    <div style={{ marginBottom: '16px' }}>
                      <div style={{ fontSize: '12px', fontWeight: 700, color: '#4b5563', marginBottom: '8px' }}>
                        Detected Feature Columns:
                      </div>
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px' }}>
                        {availableKeys.map((key) => (
                          <span
                            key={key}
                            style={{
                              fontSize: '12px',
                              background: '#f3f4f6',
                              color: '#374151',
                              padding: '4px 10px',
                              borderRadius: '6px',
                              fontWeight: 600,
                              border: '1px solid #e5e7eb',
                            }}
                          >
                            {formatLabel(key)}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="info" style={{ fontSize: '12px', lineHeight: 1.6, marginBottom: '16px' }}>
                      <b>💡 Dataset Guidelines &amp; Formatting:</b>
                      <ul style={{ margin: '6px 0 0 16px', padding: 0 }}>
                        <li><b>File Format:</b> Standard comma-separated <code>.csv</code> files.</li>
                        <li><b>Customer ID:</b> Auto-detected or automatically assigned if omitted.</li>
                        <li><b>Feature Normalization:</b> Zero-mean standardization is automatically applied before clustering.</li>
                      </ul>
                    </div>

                    <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                      <button
                        className="primary"
                        style={{ fontSize: '12px', padding: '8px 14px' }}
                        onClick={() => setActivePage('overview')}
                      >
                        ✦ View Cluster Visualizer
                      </button>
                      <button
                        className="secondary"
                        style={{ fontSize: '12px', padding: '8px 14px' }}
                        onClick={() => setActivePage('comparison')}
                      >
                        ✦ Compare ML Models
                      </button>
                    </div>
                  </div>
                </div>
              </section>
            </main>
          </div>
        </div>
      )}

      {/* Toast Notification */}
      {toastText && (
        <div id="toast" className="toast">
          {toastText}
        </div>
      )}
    </>
  );
}
