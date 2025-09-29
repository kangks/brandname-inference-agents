# Test Migration Plan

## Overview

This document outlines the detailed migration plan for converting existing test files to pytest framework format as part of task 6.1.

## Migration Strategy

### 1. Preserve Existing Functionality
- Maintain all existing test logic and assertions
- Preserve test coverage for all components
- Ensure backward compatibility during transition

### 2. Improve Test Organization
- Follow pytest discovery patterns
- Use consistent naming conventions
- Implement proper test categorization with markers

### 3. Enhance Test Quality
- Add proper fixtures and test data management
- Implement consistent error handling
- Improve test isolation and cleanup

## File-by-File Migration Plan

### High Priority Files (Core Functionality)

#### 1. `test_orchestrator_coordination.py` → `integration/test_orchestrator_coordination.py`

**Current Structure:**
- Script-based with `asyncio.run(main())`
- Manual test execution and reporting
- Direct agent instantiation and testing

**Migration Actions:**
- Convert to pytest class-based structure
- Use `@pytest.mark.asyncio` for async tests
- Implement proper fixtures for orchestrator setup
- Add parametrized tests for different products
- Use assertion helpers for result validation

**New Structure:**
```python
@pytest.mark.integration
@pytest.mark.asyncio
class TestOrchestratorCoordination:
    async def test_orchestrator_agent_coordination(self, orchestrator_instance, sample_product_input):
        # Converted test logic
        pass
```

#### 2. `test_orchestrator_api.py` → `end_to_end/test_api_endpoints.py`

**Current Structure:**
- Class-based API tester with manual execution
- HTTP requests testing with manual assertions
- Product-specific test cases

**Migration Actions:**
- Convert to pytest test functions
- Use `requests_mock` or similar for API testing
- Implement API client fixture
- Add parametrized tests for different endpoints
- Use proper HTTP status code assertions

**New Structure:**
```python
@pytest.mark.e2e
@pytest.mark.api
class TestAPIEndpoints:
    def test_health_endpoint(self, api_client):
        # Converted test logic
        pass
    
    @pytest.mark.parametrize("product_name,expected_brand", [
        ("Samsung Galaxy S24", "Samsung"),
        ("iPhone 15", "Apple"),
    ])
    def test_inference_endpoint(self, api_client, product_name, expected_brand):
        # Converted test logic
        pass
```

#### 3. `test_inference_system.py` → `end_to_end/test_system_integration.py`

**Current Structure:**
- Async script with multiple test functions
- Manual test execution and result aggregation
- System-wide integration testing

**Migration Actions:**
- Convert to pytest class structure
- Use proper async fixtures
- Implement system-wide setup/teardown
- Add comprehensive assertions
- Use test markers for categorization

#### 4. `final_comprehensive_test.py` → `end_to_end/test_comprehensive_methods.py`

**Current Structure:**
- Script-based method testing
- Manual HTTP requests and result parsing
- Method comparison and ranking

**Migration Actions:**
- Convert to pytest parametrized tests
- Use API client fixture
- Implement method comparison utilities
- Add proper assertions for each method
- Use test data factory for test cases

### Medium Priority Files (Method Testing)

#### 1. `test_method_selection.py` → `integration/test_method_selection.py`

**Migration Actions:**
- Convert HTTP requests to use API client fixture
- Add parametrized tests for each method
- Implement proper error handling assertions
- Use test markers for method-specific tests

#### 2. `test_ner_llm_methods.py` → `integration/test_method_specific.py`

**Migration Actions:**
- Convert to pytest structure
- Use agent fixtures for testing
- Add parametrized tests for different inputs
- Implement proper result validation

#### 3. `test_simple_inference_working.py` → `unit/test_simple_inference.py`

**Migration Actions:**
- Convert async script to pytest async tests
- Use agent fixtures and mock configurations
- Add parametrized tests for multilingual support
- Implement proper cleanup in fixtures

### Low Priority Files (Consolidation)

#### Custom Deployment Tests Consolidation

**Files to Consolidate:**
- `test_custom_deployment_direct.py`
- `test_custom_deployment_final.py`
- `test_custom_deployment_fix.py`
- `test_custom_deployment_local.py`
- `test_custom_deployment_working.py`
- `test_orchestrator_custom_deployment.py`
- `test_orchestrator_custom_deployment_fixed.py`

**Target:** `end_to_end/test_custom_deployments.py`

**Migration Actions:**
- Extract common functionality into fixtures
- Create parametrized tests for different deployment scenarios
- Implement proper deployment configuration management
- Add comprehensive error handling and cleanup

#### Validation Scripts Conversion

**Files to Convert:**
- `final_validation.py` → Add assertions to existing E2E tests
- `validate_inference_api_fixed.py` → `end_to_end/test_api_validation.py`
- `validate_orchestrator_deployment.py` → `end_to_end/test_deployment_validation.py`
- `validate_strands_multiagent.py` → `integration/test_strands_validation.py`

## Required Fixtures and Utilities

### New Fixtures Needed

#### API Testing Fixtures
```python
@pytest.fixture
def api_client(test_config):
    """HTTP client for API testing."""
    pass

@pytest.fixture
def api_base_url(test_config):
    """Base URL for API endpoints."""
    pass
```

#### Orchestrator Testing Fixtures
```python
@pytest.fixture
async def orchestrator_instance(test_config):
    """Orchestrator instance for coordination testing."""
    pass

@pytest.fixture
def coordination_test_data():
    """Test data for coordination testing."""
    pass
```

#### System Testing Fixtures
```python
@pytest.fixture(scope="session")
def system_test_config():
    """System-wide test configuration."""
    pass

@pytest.fixture
def deployment_config():
    """Deployment configuration for testing."""
    pass
```

### New Test Utilities

#### API Test Helpers
```python
class APITestHelpers:
    @staticmethod
    def assert_successful_response(response, expected_fields=None):
        """Assert API response is successful."""
        pass
    
    @staticmethod
    def assert_error_response(response, expected_status_code, expected_error_pattern=None):
        """Assert API response contains expected error."""
        pass
```

#### Coordination Test Helpers
```python
class CoordinationTestHelpers:
    @staticmethod
    def assert_orchestrator_coordination(result, expected_agents=None):
        """Assert orchestrator coordination results."""
        pass
    
    @staticmethod
    def validate_agent_results(agent_results, minimum_successful=1):
        """Validate individual agent results."""
        pass
```

## Test Markers Implementation

### Add New Markers to pytest.ini
```ini
[tool:pytest]
markers =
    unit: Unit tests for individual components
    integration: Integration tests for component interactions
    e2e: End-to-end tests for complete system
    aws: Tests requiring AWS environment
    api: Tests for API endpoints
    coordination: Tests for orchestrator coordination
    slow: Long-running tests
    custom_deployment: Tests for custom deployment scenarios
```

### Marker Usage Guidelines
- `@pytest.mark.unit` - Fast, isolated tests with mocked dependencies
- `@pytest.mark.integration` - Component interaction tests with minimal external dependencies
- `@pytest.mark.e2e` - Full system tests with actual services
- `@pytest.mark.aws` - Tests requiring AWS credentials and services
- `@pytest.mark.api` - HTTP API endpoint tests
- `@pytest.mark.coordination` - Orchestrator coordination tests
- `@pytest.mark.slow` - Tests taking >5 seconds
- `@pytest.mark.custom_deployment` - Custom deployment scenario tests

## Migration Timeline

### Week 1: High Priority Files
- **Day 1-2:** Convert orchestrator coordination and API tests
- **Day 3:** Convert system integration tests
- **Day 4:** Convert comprehensive method tests
- **Day 5:** Testing and validation of converted tests

### Week 2: Medium Priority Files
- **Day 1:** Convert method selection and specific method tests
- **Day 2:** Convert simple inference tests
- **Day 3:** Implement missing fixtures and utilities
- **Day 4:** Testing and integration of medium priority tests
- **Day 5:** Buffer for issues and refinements

### Week 3: Low Priority Files and Cleanup
- **Day 1-2:** Consolidate custom deployment tests
- **Day 3:** Convert validation scripts
- **Day 4:** Remove obsolete test files and cleanup
- **Day 5:** Final validation and documentation

## Quality Assurance

### Conversion Validation
1. **Functionality Preservation:** All original test logic must be preserved
2. **Coverage Maintenance:** Test coverage must not decrease
3. **Performance:** Converted tests should run faster or same speed
4. **Reliability:** Tests must be more reliable and less flaky

### Testing the Tests
1. **Run Original Tests:** Ensure original tests still pass before conversion
2. **Run Converted Tests:** Verify converted tests pass with same results
3. **Compare Results:** Ensure converted tests produce equivalent results
4. **Edge Case Testing:** Test converted tests with edge cases and failures

### Documentation Requirements
1. **Migration Notes:** Document any changes in test behavior
2. **Usage Instructions:** Update test execution documentation
3. **Troubleshooting:** Document common issues and solutions
4. **Maintenance Guide:** Instructions for maintaining converted tests

## Success Criteria

### Technical Success
- [ ] All existing test functionality preserved
- [ ] Tests follow pytest conventions and best practices
- [ ] Proper test organization and discovery
- [ ] Consistent test data and mocking strategies
- [ ] Improved error handling and reporting
- [ ] Better test isolation and cleanup

### Process Success
- [ ] Migration completed within timeline
- [ ] No regression in test coverage
- [ ] Improved test execution speed
- [ ] Enhanced test reliability
- [ ] Better developer experience

### Documentation Success
- [ ] Comprehensive migration documentation
- [ ] Updated test execution guides
- [ ] Clear troubleshooting instructions
- [ ] Maintenance and contribution guidelines

## Risk Mitigation

### Technical Risks
- **Test Behavior Changes:** Maintain detailed comparison logs
- **Performance Regression:** Benchmark before and after conversion
- **Dependency Issues:** Test in isolated environments
- **Integration Failures:** Gradual migration with rollback capability

### Process Risks
- **Timeline Delays:** Buffer time built into schedule
- **Resource Constraints:** Prioritized migration order
- **Quality Issues:** Comprehensive validation process
- **Knowledge Transfer:** Detailed documentation and reviews