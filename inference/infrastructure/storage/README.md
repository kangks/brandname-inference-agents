# Milvus Storage Configuration

This directory contains configuration files for different Milvus storage options in the multilingual inference system.

## Storage Options

### 1. Local Disk Storage (Default)

**Configuration**: `milvus-local.yaml`
**Task Definition**: `milvus-local-storage-task-def.json`

**Characteristics**:
- ✅ **Faster performance** - Direct disk I/O
- ✅ **Lower cost** - No additional storage charges
- ✅ **Simpler setup** - No external dependencies
- ❌ **Ephemeral** - Data lost when container restarts
- ❌ **Not shared** - Each container has its own data

**Best for**:
- Development and testing environments
- Stateless applications
- Cost-sensitive deployments
- High-performance requirements

**Usage**:
```bash
# Deploy with local storage (default)
STORAGE_TYPE=local ./deploy-milvus.sh
```

### 2. EFS Storage (Optional)

**Configuration**: `efs-milvus.json`, `milvus.yaml`
**Task Definition**: `milvus-standalone-task-def.json`, `milvus-task-def.json`
**CloudFormation**: `efs-storage.yaml`

**Characteristics**:
- ✅ **Persistent** - Data survives container restarts
- ✅ **Shared** - Multiple containers can access same data
- ✅ **Scalable** - Grows automatically with usage
- ✅ **Production-ready** - Suitable for production workloads
- ❌ **Slightly slower** - Network-attached storage
- ❌ **Additional cost** - EFS storage and throughput charges

**Best for**:
- Production environments
- Multi-container deployments
- Data persistence requirements
- Shared storage needs

**Usage**:
```bash
# Deploy with EFS storage
STORAGE_TYPE=efs ./deploy-milvus.sh
```

## File Structure

```
storage/
├── README.md                    # This documentation
├── efs-milvus.json             # EFS configuration (legacy)
└── ../cloudformation/
    └── efs-storage.yaml        # CloudFormation template for EFS
```

## Configuration Files

### Local Storage Configuration

The local storage configuration uses the container's ephemeral storage:

```yaml
# milvus-local.yaml
localStorage:
  path: /var/lib/milvus/data

storage:
  type: local
  autoFlush: true
  flushInterval: 60
```

### EFS Storage Configuration

The EFS configuration provides persistent, shared storage:

```json
{
  "fileSystemConfiguration": {
    "performanceMode": "generalPurpose",
    "throughputMode": "provisioned",
    "provisionedThroughputInMibps": 100,
    "encrypted": true
  }
}
```

## Deployment Scripts

### New Unified Deployment Script

Use the new `deploy-milvus.sh` script for flexible storage configuration:

```bash
# Show help and options
./deploy-milvus.sh --help

# Deploy with local storage (default)
./deploy-milvus.sh
STORAGE_TYPE=local ./deploy-milvus.sh

# Deploy with EFS storage
STORAGE_TYPE=efs ./deploy-milvus.sh

# Deploy distributed mode with EFS
STORAGE_TYPE=efs MILVUS_MODE=distributed ./deploy-milvus.sh
```

### Environment Variables

- `STORAGE_TYPE`: `local` (default) or `efs`
- `MILVUS_MODE`: `standalone` (default) or `distributed`
- `AWS_PROFILE`: AWS profile to use (default: `ml-sandbox`)
- `AWS_REGION`: AWS region (default: `us-east-1`)

## Migration Guide

### From EFS to Local Storage

1. **Backup data** (if needed):
   ```bash
   # Connect to running EFS-based container
   aws ecs execute-command --cluster multilingual-inference-cluster \
     --task <task-id> --container milvus-standalone --interactive --command "/bin/bash"
   
   # Export collections or backup data as needed
   ```

2. **Deploy with local storage**:
   ```bash
   STORAGE_TYPE=local ./deploy-milvus.sh
   ```

3. **Reimport data** (if needed):
   - Use Milvus client to recreate collections
   - Reimport vector data

### From Local to EFS Storage

1. **Deploy EFS infrastructure**:
   ```bash
   STORAGE_TYPE=efs ./deploy-milvus.sh
   ```

2. **Migrate data** (if needed):
   - Export data from local storage before switching
   - Import data to EFS-backed Milvus instance

## Cost Considerations

### Local Storage
- **Cost**: Included in ECS Fargate pricing
- **Performance**: Highest (local SSD)
- **Durability**: Low (ephemeral)

### EFS Storage
- **Cost**: EFS storage ($0.30/GB/month) + throughput charges
- **Performance**: Good (network-attached)
- **Durability**: High (99.999999999% durability)

## Performance Comparison

| Metric | Local Storage | EFS Storage |
|--------|---------------|-------------|
| Read Latency | ~1ms | ~3-5ms |
| Write Latency | ~1ms | ~5-10ms |
| Throughput | High | Configurable |
| IOPS | High | Up to 7,000 |
| Durability | Low | Very High |
| Sharing | No | Yes |

## Troubleshooting

### Local Storage Issues

1. **Container out of space**:
   - Increase ECS task memory allocation
   - Implement data cleanup policies
   - Monitor disk usage

2. **Data loss on restart**:
   - Expected behavior with local storage
   - Consider switching to EFS for persistence

### EFS Storage Issues

1. **Slow performance**:
   - Check provisioned throughput settings
   - Verify network connectivity
   - Monitor EFS CloudWatch metrics

2. **Mount failures**:
   - Verify security group rules (port 2049)
   - Check EFS access points configuration
   - Ensure proper IAM permissions

3. **High costs**:
   - Monitor EFS usage in CloudWatch
   - Adjust provisioned throughput
   - Consider EFS Intelligent Tiering

## Monitoring

### CloudWatch Metrics

**Local Storage**:
- ECS task CPU/Memory utilization
- Container disk usage (if available)

**EFS Storage**:
- `AWS/EFS` namespace metrics
- `TotalIOBytes`, `DataReadIOBytes`, `DataWriteIOBytes`
- `PercentIOLimit`, `BurstCreditBalance`

### Logging

Both configurations use CloudWatch Logs:
- Log Group: `/ecs/multilingual-inference-milvus`
- Monitor for storage-related errors
- Check Milvus startup logs for configuration issues

## Best Practices

### Local Storage
1. **Monitor disk usage** regularly
2. **Implement data lifecycle** policies
3. **Use for stateless** workloads
4. **Plan for data loss** scenarios

### EFS Storage
1. **Right-size throughput** provisioning
2. **Use access points** for security
3. **Enable encryption** at rest and in transit
4. **Monitor costs** regularly
5. **Implement backup** strategies

## Support

For issues with storage configuration:

1. Check the deployment logs
2. Verify ECS task definitions
3. Review CloudWatch metrics
4. Consult the main documentation:
   - `../docs/ARCHITECTURE_AND_DEPLOYMENT_GUIDE.md`
   - `../../INFERENCE_ARCHITECTURE_FAQ.md`