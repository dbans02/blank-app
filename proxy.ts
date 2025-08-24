import express, { Request, Response, NextFunction } from 'express';
import Redis from 'ioredis';
import rateLimit from 'express-rate-limit';

const redis = new Redis('redis://localhost:6379');
const app = express();

// Rate limiting middleware
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP'
});

app.use(limiter);

// Cache middleware
const cache = (duration: number) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    const key = `cache:${req.originalUrl}`;
    const cached = await redis.get(key);
    
    if (cached) {
      return res.json(JSON.parse(cached));
    }
    
    res.sendResponse = res.json;
    res.json = (body: any) => {
      redis.setex(key, duration, JSON.stringify(body));
      res.sendResponse(body);
    };
    
    next();
  };
};

// User service proxy
app.get('/api/users/:id', cache(300), async (req: Request, res: Response) => {
  const userId = req.params.id;
  
  try {
    const response = await fetch(`http://user-service:3000/users/${userId}`);
    const user = await response.json();
    
    if (!response.ok) {
      return res.status(response.status).json(user);
    }
    
    return res.json(user);
  } catch (error) {
    return res.status(500).json({ error: 'Service unavailable' });
  }
});

// Product search with caching
app.get('/api/products/search', cache(600), async (req: Request, res: Response) => {
  const { query, category, limit = 10, offset = 0 } = req.query;
  
  const searchKey = `search:${query}:${category}:${limit}:${offset}`;
  
  try {
    const cachedResults = await redis.get(searchKey);
    if (cachedResults) {
      return res.json(JSON.parse(cachedResults));
    }
    
    const response = await fetch(
      `http://product-service:3000/search?query=${query}&category=${category}&limit=${limit}&offset=${offset}`
    );
    
    const results = await response.json();
    
    // Cache search results for 10 minutes
    await redis.setex(searchKey, 600, JSON.stringify(results));
    
    return res.json(results);
  } catch (error) {
    return res.status(500).json({ error: 'Search service unavailable' });
  }
});

// Health check endpoint
app.get('/health', async (req: Request, res: Response) => {
  const services = ['user-service', 'product-service', 'order-service'];
  const healthChecks = [];
  
  for (const service of services) {
    try {
      const response = await fetch(`http://${service}:3000/health`, {
        timeout: 5000
      });
      healthChecks.push({
        service,
        status: response.ok ? 'healthy' : 'unhealthy',
        responseTime: response.headers.get('response-time')
      });
    } catch (error) {
      healthChecks.push({
        service,
        status: 'unhealthy',
        error: error.message
      });
    }
  }
  
  const allHealthy = healthChecks.every(check => check.status === 'healthy');
  
  return res.status(allHealthy ? 200 : 503).json({
    status: allHealthy ? 'healthy' : 'degraded',
    services: healthChecks
  });
});

app.listen(3000, () => {
  console.log('API Gateway running on port 3000');
});
