from agents.actor_critic_agents.A3C import A3C

class A2C(A3C):
    """Synchronous version of A2C algorithm from deepmind paper https://arxiv.org/pdf/1602.01783.pdf. The only
    difference between this and the A3C is that gradient updates get done in a batch rather than 1 by 1 as the gradients
    come in"""
    agent_name = "A2C"
    def __init__(self, config):
        super(A2C, self).__init__(config)

    def update_shared_model(self, gradient_updates_queue):
        """Worker that updates the shared model with gradients as they get put into the queue"""
        while True:
            gradients_seen = 0
            while gradients_seen < self.worker_processes:
                if gradients_seen == 0:
                    gradients = gradient_updates_queue.get()
                else:
                    new_grads = gradient_updates_queue.get()
                    gradients = [grad + new_grad for grad, new_grad in zip(gradients, new_grads)]
                gradients_seen += 1
            self.actor_critic_optimizer.zero_grad()
            for grads, params in zip(gradients, self.actor_critic.parameters()):
                params._grad = grads
            self.actor_critic_optimizer.step()