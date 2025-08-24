import { getConnection } from 'typeorm';
import { Order } from '../entities/Order';
import { OrderItem } from '../entities/OrderItem';
import { Product } from '../entities/Product';
import { Customer } from '../entities/Customer';
import { Inventory } from '../entities/Inventory';

export class OrderService {
  async processOrder(customerId: number, items: { productId: number, quantity: number }[]) {
    const customer = await Customer.findOne(customerId);
    
    if (!customer) {
      throw new Error('Customer not found');
    }
    
    // Create order
    const order = new Order();
    order.customerId = customerId;
    order.status = 'pending';
    order.createdAt = new Date();
    await order.save();
    
    let totalAmount = 0;
    
    // Process each item
    for (const item of items) {
      const product = await Product.findOne(item.productId);
      
      if (!product) {
        throw new Error(`Product ${item.productId} not found`);
      }
      
      // Check inventory
      const inventory = await Inventory.findOne({ 
        where: { productId: item.productId }
      });
      
      if (!inventory || inventory.quantity < item.quantity) {
        throw new Error(`Insufficient inventory for product ${product.name}`);
      }
      
      // Update inventory
      inventory.quantity -= item.quantity;
      await inventory.save();
      
      // Create order item
      const orderItem = new OrderItem();
      orderItem.orderId = order.id;
      orderItem.productId = item.productId;
      orderItem.quantity = item.quantity;
      orderItem.price = product.price;
      orderItem.total = product.price * item.quantity;
      await orderItem.save();
      
      totalAmount += orderItem.total;
      
      // Send inventory alert if low
      if (inventory.quantity < 10) {
        this.sendLowInventoryAlert(product.name, inventory.quantity);
      }
    }
    
    // Update order total
    order.totalAmount = totalAmount;
    order.status = 'confirmed';
    await order.save();
    
    // Send confirmation email
    await this.sendOrderConfirmation(customer.email, order.id);
    
    return order;
  }
  
  async sendLowInventoryAlert(productName: string, quantity: number) {
    // Email sending logic
    console.log(`Low inventory alert: ${productName} - ${quantity} remaining`);
  }
  
  async sendOrderConfirmation(email: string, orderId: number) {
    // Email sending logic
    console.log(`Order confirmation sent to ${email} for order ${orderId}`);
  }
}
