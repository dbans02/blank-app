import { Connection, Channel, ConsumeMessage } from 'amqplib';
import { EventEmitter } from 'events';

interface OrderEvent {
  type: 'order.created' | 'order.updated' | 'order.cancelled';
  orderId: string;
  customerId: string;
  data: any;
  timestamp: Date;
}

export class OrderEventProcessor extends EventEmitter {
  private connection: Connection;
  private channel: Channel;
  private readonly queueName = 'order-events';
  private readonly exchangeName = 'orders';
  
  constructor(connection: Connection) {
    super();
    this.connection = connection;
  }
  
  async initialize() {
    this.channel = await this.connection.createChannel();
    
    await this.channel.assertExchange(this.exchangeName, 'topic', { durable: true });
    await this.channel.assertQueue(this.queueName, { durable: true });
    await this.channel.bindQueue(this.queueName, this.exchangeName, 'order.*');
    
    await this.channel.consume(this.queueName, this.handleMessage.bind(this));
  }
  
  private async handleMessage(msg: ConsumeMessage | null) {
    if (!msg) return;
    
    try {
      const event: OrderEvent = JSON.parse(msg.content.toString());
      
      console.log(`Processing event: ${event.type} for order ${event.orderId}`);
      
      switch (event.type) {
        case 'order.created':
          await this.handleOrderCreated(event);
          break;
        case 'order.updated':
          await this.handleOrderUpdated(event);
          break;
        case 'order.cancelled':
          await this.handleOrderCancelled(event);
          break;
        default:
          console.log(`Unknown event type: ${event.type}`);
      }
      
      this.channel.ack(msg);
      
    } catch (error) {
      console.error('Error processing message:', error);
      
      // Retry logic
      const retryCount = (msg.properties.headers['x-retry-count'] || 0) + 1;
      
      if (retryCount <= 3) {
        setTimeout(() => {
          this.channel.publish(
            this.exchangeName,
            msg.fields.routingKey,
            msg.content,
            { 
              headers: { 
                'x-retry-count': retryCount,
                'x-original-timestamp': msg.properties.headers['x-original-timestamp'] || new Date().toISOString()
              }
            }
          );
          this.channel.ack(msg);
        }, Math.pow(2, retryCount) * 1000); // Exponential backoff
      } else {
        // Send to dead letter queue
        await this.channel.publish('dead-letter', 'failed-orders', msg.content);
        this.channel.ack(msg);
        console.error(`Message failed after 3 retries, sent to DLQ`);
      }
    }
  }
  
  private async handleOrderCreated(event: OrderEvent) {
    // Send welcome email
    await this.sendEmail(event.customerId, 'order-created', event.data);
    
    // Update analytics
    await this.updateAnalytics('order_created', event.data);
    
    // Create shipping label
    if (event.data.shippingRequired) {
      await this.createShippingLabel(event.orderId, event.data.shippingAddress);
    }
  }
  
  private async handleOrderUpdated(event: OrderEvent) {
    // Notify customer
    await this.sendEmail(event.customerId, 'order-updated', event.data);
    
    // Update tracking
    if (event.data.trackingNumber) {
      await this.updateOrderTracking(event.orderId, event.data.trackingNumber);
    }
  }
  
  private async handleOrderCancelled(event: OrderEvent) {
    // Process refund
    await this.processRefund(event.orderId, event.data.refundAmount);
    
    // Cancel shipping
    if (event.data.shippingId) {
      await this.cancelShipping(event.data.shippingId);
    }
    
    // Restore inventory
    await this.restoreInventory(event.data.items);
  }
  
  private async sendEmail(customerId: string, template: string, data: any) {
    // Email service call
    console.log(`Sending ${template} email to customer ${customerId}`);
  }
  
  private async updateAnalytics(eventType: string, data: any) {
    // Analytics service call
    console.log(`Updating analytics for ${eventType}`);
  }
  
  private async createShippingLabel(orderId: string, address: any) {
    // Shipping service call
    console.log(`Creating shipping label for order ${orderId}`);
  }
  
  private async updateOrderTracking(orderId: string, trackingNumber: string) {
    // Tracking service call
    console.log(`Updating tracking for order ${orderId}: ${trackingNumber}`);
  }
  
  private async processRefund(orderId: string, amount: number) {
    // Payment service call
    console.log(`Processing refund for order ${orderId}: $${amount}`);
  }
  
  private async cancelShipping(shippingId: string) {
    // Shipping service call
    console.log(`Cancelling shipping ${shippingId}`);
  }
  
  private async restoreInventory(items: any[]) {
    // Inventory service call
    console.log(`Restoring inventory for ${items.length} items`);
  }
}
