# Hardware Frontend Implementation Plan

## 1. User Experience Design

### 1.1 Navigation Structure
```
Main App
├── Home (Dashboard)
│   └── Featured Hardware Section
├── Models
│   └── Model Analysis with Hardware Recommendations
├── Hardware (New Tab)
│   ├── Hardware List
│   └── Hardware Details
└── About
```

### 1.2 User Flows

#### Flow 1: Browse Hardware
1. User clicks "Hardware" tab in main navigation
2. Views paginated list of hardware with filters
3. Clicks "See More" on a hardware card
4. Views detailed hardware specifications
5. Sees compatible models and performance metrics

#### Flow 2: Model to Hardware
1. User analyzes a model
2. System shows recommended hardware
3. User clicks on recommended hardware
4. Views hardware details and compatibility

#### Flow 3: Hardware to Models
1. User views hardware details
2. Scrolls to "Compatible Models" section
3. Views performance table for different configurations
4. Can navigate to specific model details

## 2. UI Components Design

### 2.1 Main Page - Featured Hardware Section
```jsx
// Location: Below hero section, above existing content
<FeaturedHardware>
  <SectionTitle>Popular AI Hardware</SectionTitle>
  <HardwareGrid>
    {/* Display 5-6 hardware cards in responsive grid */}
    <HardwareCard>
      <CardImage src="/api/hardware/logo/{name}" />
      <CardTitle>NVIDIA H100</CardTitle>
      <QuickSpecs>
        <Spec>80GB Memory</Spec>
        <Spec>3.35 TB/s Bandwidth</Spec>
        <Spec>1,979 TFLOPS</Spec>
      </QuickSpecs>
      <CardAction onClick={navigateToDetails}>View Details →</CardAction>
    </HardwareCard>
  </HardwareGrid>
  <ViewAllButton href="/hardware">View All Hardware →</ViewAllButton>
</FeaturedHardware>
```

### 2.2 Hardware List Page
```jsx
// Route: /hardware
<HardwareListPage>
  <PageHeader>
    <Title>AI Hardware Catalog</Title>
    <Subtitle>Explore GPUs, TPUs, and accelerators for AI workloads</Subtitle>
  </PageHeader>
  
  <FilterSection>
    <SearchBar placeholder="Search hardware..." />
    <FilterGroup>
      <Select label="Type" options={['All', 'GPU', 'TPU', 'CPU', 'Accelerator']} />
      <Select label="Manufacturer" options={['All', 'NVIDIA', 'AMD', 'Intel', 'Google']} />
      <RangeSlider label="Memory (GB)" min={0} max={256} />
      <RangeSlider label="Performance (TFLOPS)" min={0} max={5000} />
      <Select label="Sort By" options={['Name', 'Performance', 'Memory', 'Price']} />
    </FilterGroup>
  </FilterSection>
  
  <HardwareGrid>
    {hardwareList.map(hw => (
      <HardwareCard key={hw.name}>
        <CardHeader>
          <Manufacturer>{hw.manufacturer}</Manufacturer>
          <Type>{hw.type}</Type>
        </CardHeader>
        <CardTitle>{hw.name}</CardTitle>
        <SpecsList>
          <SpecItem icon="memory">Memory: {hw.memory_size} GB</SpecItem>
          <SpecItem icon="speed">Performance: {hw.flops} TFLOPS</SpecItem>
          <SpecItem icon="bandwidth">Bandwidth: {hw.memory_bw} GB/s</SpecItem>
          <SpecItem icon="power">Power: {hw.power || 'N/A'} W</SpecItem>
        </SpecsList>
        <PriceRange>
          {hw.min_on_prem_price && (
            <Price>From ${hw.min_on_prem_price.toLocaleString()}</Price>
          )}
        </PriceRange>
        <CardActions>
          <Button variant="primary" onClick={() => navigate(`/hardware/${hw.name}`)}>
            See More
          </Button>
        </CardActions>
      </HardwareCard>
    ))}
  </HardwareGrid>
  
  <Pagination>
    <PageInfo>Showing {start}-{end} of {total} items</PageInfo>
    <PageControls>
      <Button disabled={!hasPrev}>Previous</Button>
      <PageNumbers>{/* 1 2 3 ... 10 */}</PageNumbers>
      <Button disabled={!hasNext}>Next</Button>
    </PageControls>
  </Pagination>
</HardwareListPage>
```

### 2.3 Hardware Detail Page
```jsx
// Route: /hardware/:hardwareName
<HardwareDetailPage>
  <BackButton onClick={() => navigate('/hardware')}>← Back to Hardware</BackButton>
  
  <HeroSection>
    <HardwareImage src={hw.image || '/default-hardware.png'} />
    <HeroContent>
      <Title>{hw.name}</Title>
      <Manufacturer>{hw.manufacturer}</Manufacturer>
      <Description>{hw.description}</Description>
      <QuickActions>
        <Button variant="primary">Compare</Button>
        <Button variant="secondary">Download Specs</Button>
      </QuickActions>
    </HeroContent>
  </HeroSection>
  
  <TabContainer>
    <Tab label="Specifications" active>
      <SpecificationsGrid>
        <SpecGroup title="Performance">
          <SpecRow>
            <Label>Compute Performance</Label>
            <Value>{hw.flops} TFLOPS</Value>
            <Tooltip>Floating-point operations per second measuring raw compute power</Tooltip>
          </SpecRow>
          <SpecRow>
            <Label>Memory Size</Label>
            <Value>{hw.memory_size} GB</Value>
            <Tooltip>Total available memory for model weights and activations</Tooltip>
          </SpecRow>
          <SpecRow>
            <Label>Memory Bandwidth</Label>
            <Value>{hw.memory_bw} GB/s</Value>
            <Tooltip>Data transfer rate between processor and memory</Tooltip>
          </SpecRow>
          <SpecRow>
            <Label>Interconnect Bandwidth</Label>
            <Value>{hw.icn || 'N/A'} GB/s</Value>
            <Tooltip>Communication speed between multiple devices</Tooltip>
          </SpecRow>
        </SpecGroup>
        
        <SpecGroup title="Power & Efficiency">
          <SpecRow>
            <Label>Power Consumption</Label>
            <Value>{hw.power || 'N/A'} W</Value>
            <Tooltip>Maximum power draw under full load</Tooltip>
          </SpecRow>
          <SpecRow>
            <Label>Performance per Watt</Label>
            <Value>{hw.power ? (hw.flops / hw.power).toFixed(2) : 'N/A'} TFLOPS/W</Value>
            <Tooltip>Energy efficiency metric for compute operations</Tooltip>
          </SpecRow>
        </SpecGroup>
      </SpecificationsGrid>
    </Tab>
    
    <Tab label="Pricing">
      <PricingSection>
        <PricingGroup title="On-Premise Vendors">
          {hw.vendors.length > 0 ? (
            <VendorList>
              {hw.vendors.map(vendor => (
                <VendorCard key={vendor.vendor_name}>
                  <VendorName>{vendor.vendor_name}</VendorName>
                  <PriceRange>
                    ${vendor.price_lower?.toLocaleString() || 'Contact'} - 
                    ${vendor.price_upper?.toLocaleString() || 'Contact'}
                  </PriceRange>
                </VendorCard>
              ))}
            </VendorList>
          ) : (
            <EmptyState>No vendor pricing available</EmptyState>
          )}
        </PricingGroup>
        
        <PricingGroup title="Cloud Providers">
          {hw.clouds.length > 0 ? (
            <CloudList>
              {hw.clouds.map(cloud => (
                <CloudCard key={cloud.cloud_name}>
                  <CloudName>{cloud.cloud_name}</CloudName>
                  <InstanceType>{cloud.instance_name}</InstanceType>
                  <HourlyRate>
                    ${cloud.price_lower} - ${cloud.price_upper}/hour
                  </HourlyRate>
                  <Regions>{cloud.regions.join(', ')}</Regions>
                </CloudCard>
              ))}
            </CloudList>
          ) : (
            <EmptyState>Not available on cloud platforms</EmptyState>
          )}
        </PricingGroup>
      </PricingSection>
    </Tab>
    
    <Tab label="Compatible Models">
      <ModelCompatibilitySection>
        <SectionHeader>
          <Title>Model Performance Estimates</Title>
          <Subtitle>Based on batch size 10 and various sequence lengths</Subtitle>
        </SectionHeader>
        
        <CompatibilityTable>
          <TableHeader>
            <Column>Model</Column>
            <Column>Parameters</Column>
            <Column>Seq 2K</Column>
            <Column>Seq 4K</Column>
            <Column>Seq 8K</Column>
            <Column>Seq 16K</Column>
            <Column>Status</Column>
          </TableHeader>
          <TableBody>
            {compatibleModels.map(model => (
              <TableRow key={model.name}>
                <Cell>{model.name}</Cell>
                <Cell>{model.parameters}B</Cell>
                <Cell>
                  <PerformanceIndicator status={model.seq2k.status}>
                    {model.seq2k.memory} GB
                  </PerformanceIndicator>
                </Cell>
                <Cell>
                  <PerformanceIndicator status={model.seq4k.status}>
                    {model.seq4k.memory} GB
                  </PerformanceIndicator>
                </Cell>
                <Cell>
                  <PerformanceIndicator status={model.seq8k.status}>
                    {model.seq8k.memory} GB
                  </PerformanceIndicator>
                </Cell>
                <Cell>
                  <PerformanceIndicator status={model.seq16k.status}>
                    {model.seq16k.memory} GB
                  </PerformanceIndicator>
                </Cell>
                <Cell>
                  <StatusBadge status={model.overallStatus} />
                </Cell>
              </TableRow>
            ))}
          </TableBody>
        </CompatibilityTable>
        
        <Legend>
          <LegendItem color="green">✓ Fits comfortably</LegendItem>
          <LegendItem color="yellow">⚠ Tight fit</LegendItem>
          <LegendItem color="red">✗ Exceeds memory</LegendItem>
        </Legend>
      </ModelCompatibilitySection>
    </Tab>
  </TabContainer>
</HardwareDetailPage>
```

### 2.4 Model Analysis - Hardware Recommendations
```jsx
// Addition to existing model analysis page
<HardwareRecommendations>
  <SectionTitle>Recommended Hardware</SectionTitle>
  <RecommendationContext>
    Based on: {modelName} • {totalMemory}GB required • Batch size: {batchSize}
  </RecommendationContext>
  
  <RecommendationGrid>
    {recommendations.map(rec => (
      <RecommendationCard key={rec.hardware_name}>
        <CardHeader>
          <HardwareName>{rec.hardware_name}</HardwareName>
          <NodeCount>{rec.nodes_required} node(s)</NodeCount>
        </CardHeader>
        <SpecHighlights>
          <Spec>Memory: {rec.memory_per_chip} GB</Spec>
          <Spec>Type: {rec.type}</Spec>
        </SpecHighlights>
        <FitIndicator>
          <ProgressBar 
            value={totalMemory} 
            max={rec.memory_per_chip * rec.nodes_required} 
          />
          <FitText>{((totalMemory / (rec.memory_per_chip * rec.nodes_required)) * 100).toFixed(0)}% utilization</FitText>
        </FitIndicator>
        <CardAction onClick={() => navigate(`/hardware/${rec.hardware_name}`)}>
          View Details →
        </CardAction>
      </RecommendationCard>
    ))}
  </RecommendationGrid>
</HardwareRecommendations>
```

## 3. API Integration Details

### 3.1 Hardware List Page
```javascript
// API: GET /api/hardware
// Used in: HardwareListPage component

const fetchHardwareList = async (filters) => {
  const params = new URLSearchParams({
    type: filters.type || null,
    manufacturer: filters.manufacturer || null,
    min_memory: filters.minMemory || null,
    max_memory: filters.maxMemory || null,
    min_flops: filters.minFlops || null,
    max_flops: filters.maxFlops || null,
    sort_by: filters.sortBy || 'name',
    sort_order: filters.sortOrder || 'asc',
    limit: filters.limit || 12,
    offset: filters.offset || 0
  });
  
  const response = await fetch(`/api/hardware?${params}`);
  return response.json();
};

// Alternative: Use filter endpoint for advanced filtering
const fetchFilteredHardware = async (filters) => {
  const params = new URLSearchParams({
    query: filters.searchQuery,
    type: filters.types, // array
    manufacturer: filters.manufacturers, // array
    min_memory: filters.minMemory,
    max_memory: filters.maxMemory,
    sort_by: filters.sortBy,
    sort_order: filters.sortOrder,
    limit: 12,
    offset: (filters.page - 1) * 12
  });
  
  const response = await fetch(`/api/hardware/filter?${params}`);
  return response.json();
};
```

### 3.2 Hardware Detail Page
```javascript
// API: GET /api/hardware/{hardware_name}
// Used in: HardwareDetailPage component

const fetchHardwareDetails = async (hardwareName) => {
  const response = await fetch(`/api/hardware/${encodeURIComponent(hardwareName)}`);
  const data = await response.json();
  
  // Returns detailed hardware info including:
  // - Basic specs (flops, memory_size, memory_bw, etc.)
  // - Vendors array with pricing
  // - Clouds array with instance types and pricing
  return data;
};
```

### 3.3 Model Compatibility Calculation
```javascript
// Client-side calculation for compatible models
// Used in: HardwareDetailPage - Compatible Models tab

const calculateModelCompatibility = (hardware, models) => {
  const batchSize = 10;
  const sequenceLengths = [2000, 4000, 8000, 16000];
  
  return models.map(model => {
    const compatibility = sequenceLengths.map(seqLen => {
      // Calculate memory requirement
      const memoryRequired = calculateMemoryRequirement(
        model.parameters,
        seqLen,
        batchSize,
        model.precision || 'fp16'
      );
      
      const status = memoryRequired <= hardware.memory_size * 0.8 ? 'green' :
                    memoryRequired <= hardware.memory_size ? 'yellow' : 'red';
      
      return {
        seqLen,
        memory: memoryRequired,
        status
      };
    });
    
    return {
      ...model,
      seq2k: compatibility[0],
      seq4k: compatibility[1],
      seq8k: compatibility[2],
      seq16k: compatibility[3],
      overallStatus: compatibility.every(c => c.status === 'green') ? 'optimal' :
                     compatibility.some(c => c.status !== 'red') ? 'partial' : 'incompatible'
    };
  });
};
```

### 3.4 Hardware Recommendations
```javascript
// API: POST /api/hardware/recommend
// Used in: Model Analysis page

const fetchHardwareRecommendations = async (modelConfig) => {
  const response = await fetch('/api/hardware/recommend', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      total_memory_gb: modelConfig.totalMemory,
      model_params_b: modelConfig.parameters
    })
  });
  
  return response.json();
};

// Integration in model analysis
const analyzeModel = async (modelData) => {
  // Existing model analysis...
  const analysis = await calculateModelMemory(modelData);
  
  // Fetch hardware recommendations
  const recommendations = await fetchHardwareRecommendations({
    totalMemory: analysis.totalMemoryGB,
    parameters: modelData.parameters
  });
  
  return {
    ...analysis,
    hardwareRecommendations: recommendations
  };
};
```

### 3.5 Main Page Featured Hardware
```javascript
// API: GET /api/hardware/filter
// Used in: Main page featured section

const fetchFeaturedHardware = async () => {
  // Fetch top hardware by performance
  const params = new URLSearchParams({
    sort_by: 'flops',
    sort_order: 'desc',
    limit: 6,
    type: ['gpu', 'accelerator'] // Focus on high-performance hardware
  });
  
  const response = await fetch(`/api/hardware/filter?${params}`);
  return response.json();
};
```

## 4. State Management

### 4.1 Redux Store Structure
```javascript
// store/slices/hardwareSlice.js
const hardwareSlice = createSlice({
  name: 'hardware',
  initialState: {
    list: {
      items: [],
      total: 0,
      loading: false,
      error: null,
      filters: {
        type: null,
        manufacturer: null,
        minMemory: null,
        maxMemory: null,
        sortBy: 'name',
        sortOrder: 'asc',
        page: 1,
        limit: 12
      }
    },
    details: {
      current: null,
      loading: false,
      error: null
    },
    recommendations: {
      items: [],
      loading: false,
      error: null
    },
    featured: {
      items: [],
      loading: false,
      error: null
    }
  },
  reducers: {
    setFilters: (state, action) => {
      state.list.filters = { ...state.list.filters, ...action.payload };
    },
    resetFilters: (state) => {
      state.list.filters = initialState.list.filters;
    }
  },
  extraReducers: (builder) => {
    // Async thunks for API calls
  }
});
```

## 5. Component Architecture

### 5.1 Component Hierarchy
```
src/
├── pages/
│   ├── Hardware/
│   │   ├── HardwareListPage.jsx
│   │   ├── HardwareDetailPage.jsx
│   │   └── index.js
├── components/
│   ├── Hardware/
│   │   ├── HardwareCard.jsx
│   │   ├── HardwareGrid.jsx
│   │   ├── HardwareFilters.jsx
│   │   ├── SpecificationTable.jsx
│   │   ├── PricingSection.jsx
│   │   ├── ModelCompatibilityTable.jsx
│   │   └── HardwareRecommendations.jsx
│   └── Common/
│       ├── Pagination.jsx
│       ├── FilterBar.jsx
│       └── TabContainer.jsx
└── hooks/
    ├── useHardware.js
    ├── useHardwareFilters.js
    └── useModelCompatibility.js
```

## 6. Performance Optimizations

### 6.1 Data Fetching
- Implement pagination with lazy loading
- Cache hardware details for 5 minutes
- Prefetch hardware details on hover
- Use React Query for caching and synchronization

### 6.2 Rendering
- Virtualize long hardware lists
- Lazy load images
- Memoize expensive calculations
- Use skeleton loaders during data fetch

## 7. Responsive Design

### 7.1 Breakpoints
- Mobile: < 768px (1 column grid)
- Tablet: 768px - 1024px (2 column grid)
- Desktop: > 1024px (3-4 column grid)

### 7.2 Mobile Adaptations
- Collapsible filters on mobile
- Swipeable tabs on detail page
- Simplified table views
- Bottom sheet for filters

## 8. Accessibility

### 8.1 ARIA Labels
- Proper labeling for all interactive elements
- Screen reader announcements for filter changes
- Keyboard navigation support
- Focus management in modals

### 8.2 Color Contrast
- WCAG AA compliance for all text
- Color-blind friendly status indicators
- Alternative text for all images

## 9. Error Handling

### 9.1 API Errors
```javascript
const ErrorBoundary = ({ error, retry }) => (
  <ErrorContainer>
    <ErrorIcon />
    <ErrorMessage>
      {error.status === 404 ? 'Hardware not found' :
       error.status === 500 ? 'Server error. Please try again.' :
       'Something went wrong'}
    </ErrorMessage>
    <RetryButton onClick={retry}>Try Again</RetryButton>
  </ErrorContainer>
);
```

### 9.2 Loading States
```javascript
const HardwareCardSkeleton = () => (
  <SkeletonCard>
    <SkeletonLine width="60%" height="24px" />
    <SkeletonLine width="80%" height="16px" />
    <SkeletonLine width="40%" height="16px" />
    <SkeletonLine width="100%" height="40px" />
  </SkeletonCard>
);
```

## 10. Testing Strategy

### 10.1 Unit Tests
- Test all API integration functions
- Test filter logic and state management
- Test compatibility calculations

### 10.2 Integration Tests
- Test complete user flows
- Test API error scenarios
- Test responsive behavior

### 10.3 E2E Tests
- Test hardware browsing flow
- Test model to hardware navigation
- Test filter persistence

## 11. Implementation Timeline

### Phase 1 (Week 1)
- Set up routing and navigation
- Implement hardware list page with basic filtering
- Create hardware card component

### Phase 2 (Week 2)
- Implement hardware detail page
- Add pricing and specification sections
- Integrate with existing API

### Phase 3 (Week 3)
- Add model compatibility table
- Implement hardware recommendations in model analysis
- Add featured hardware to main page

### Phase 4 (Week 4)
- Polish UI and animations
- Implement caching and performance optimizations
- Complete testing and bug fixes

## 12. Future Enhancements

### 12.1 Advanced Features
- Hardware comparison tool
- Cost calculator for cloud deployments
- Performance benchmarks visualization
- Hardware availability notifications

### 12.2 Integration Points
- Integration with deployment workflows
- Hardware reservation system
- Performance monitoring dashboards
- Cost optimization recommendations

// Hardware type definitions for BudSimulator

export interface Hardware {
  name: string;
  type: 'gpu' | 'cpu' | 'accelerator' | 'asic';
  manufacturer: string | null;
  flops: number;
  memory_size: number;
  memory_bw: number;
  icn: number | null;
  icn_ll: number | null;
  power: number | null;
  real_values: boolean;
  url: string | null;
  description: string | null;
  on_prem_vendors: string[];
  clouds: string[];
  min_on_prem_price: number | null;
  max_on_prem_price: number | null;
  source: string;
}

export interface HardwareDetail extends Hardware {
  vendors: VendorPricing[];
  clouds: CloudPricing[];
}

export interface VendorPricing {
  vendor_name: string;
  price_lower: number | null;
  price_upper: number | null;
}

export interface CloudPricing {
  cloud_name: string;
  instance_name: string;
  price_lower: number | null;
  price_upper: number | null;
  regions: string[];
}

export interface HardwareRecommendation {
  hardware_name: string;
  nodes_required: number;
  memory_per_chip: number;
  manufacturer: string | null;
  type: string;
}

export interface HardwareFilters {
  type: string | null;
  manufacturer: string | null;
  minMemory: number | null;
  maxMemory: number | null;
  minFlops: number | null;
  maxFlops: number | null;
  sortBy: string;
  sortOrder: 'asc' | 'desc';
  page: number;
  limit: number;
}

export interface ModelCompatibility {
  model_id: string;
  model_name: string;
  parameters: number;
  seq2k: CompatibilityResult;
  seq4k: CompatibilityResult;
  seq8k: CompatibilityResult;
  seq16k: CompatibilityResult;
  overallStatus: 'optimal' | 'partial' | 'incompatible';
}

export interface CompatibilityResult {
  seqLen: number;
  memory: number;
  status: 'green' | 'yellow' | 'red';
} 